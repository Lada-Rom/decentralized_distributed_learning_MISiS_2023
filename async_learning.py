from __future__ import print_function

from bluefog.common import topology_util
import bluefog.torch as bf

import os
import sys
import time
import math
import json

import warnings
warnings.simplefilter('ignore')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


class ParamLoader:
    def __init__(self, config_name):
        with open(os.path.join(os.getcwd(), os.path.dirname(__file__), config_name)) as f:
            config_obj = json.load(f)

        self.train_batch_size   = config_obj["train_batch_size"]
        self.test_batch_size    = config_obj["test_batch_size"]
        self.train_lr           = config_obj["train_lr"]
        self.train_epochs       = config_obj["train_epochs"]

        self.topo_enable_dynamic    = config_obj["topo_enable_dynamic"]
        self.topo_static_kind       = config_obj["topo_static_kind"]

        self.accum_need_ready_neighs_frac   = config_obj["accum_need_ready_neighs_frac"]
        self.accum_func_params              = config_obj["accum_func_params"]
        self.accum_initial_self_weight      = config_obj["accum_initial_self_weight"]
        
        self.log_interval   = config_obj["log_interval"]
        self.log_topo       = config_obj["log_topo"]

        self.tensor_time_event_name = config_obj["tensor_time_event_name"]
        self.tensor_stop_train_name = config_obj["tensor_stop_train_name"]


class Dataset:
    def __init__(self, params):
        data_folder_loc = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..")
        
        # train dataset
        self.train_dataset = datasets.MNIST(
            os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=bf.size(), rank=bf.rank())
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=params.train_batch_size,
            sampler=self.train_sampler, **{})

        # test dataset
        self.test_dataset = datasets.MNIST(
            os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )

        self.test_sampler = None
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_dataset, num_replicas=bf.size(), rank=bf.rank())
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=params.test_batch_size,
            sampler=self.test_sampler, **{})


class Topo:
    def __init__(self, params):
        self.dynamic_topo = None

        if params.topo_enable_dynamic is True:
            self.dynamic_topo = topology_util.\
                GetDynamicSendRecvRanks(bf.load_topology(), bf.rank())
        else:
            # 0 = Fully Connected
            # 1 = TODO
            if params.topo_static_kind == 0:
                bf.set_topology(topology_util.FullyConnectedGraph(bf.size()))

    def get_next_neights(self):
        if params.topo_enable_dynamic is True:
            return next(self.dynamic_topo)
        else:
            return bf.out_neighbor_ranks(), bf.in_neighbor_ranks()

class Net(nn.Module):
    def __init__(self, params):
        self.func_a = params.accum_func_params[0]
        self.func_b = params.accum_func_params[1]
        self.func_c = params.accum_func_params[2]
        self.self_weight = params.accum_initial_self_weight

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # time tensor: [0] for timestamp, [1] for event number
        self.name_time = params.tensor_time_event_name
        self.event_number = 0

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)
    
    def win_create_weights(self):
        for name, tensor in sorted(self.state_dict().items()):
            bf.win_create(tensor, name=name, zero_init=True)
        timestamp = torch.FloatTensor([time.time(), self.event_number])
        bf.win_create(timestamp, name=self.name_time)

    def win_put_weights(self):
        self.event_number += 1
        for name, tensor in sorted(self.state_dict().items()):
            bf.win_put(tensor, name=name)
        timestamp = torch.FloatTensor([time.time(), self.event_number])
        bf.win_put(timestamp, name=self.name_time)

    def win_update_weights(self, in_neighbors):
        dst_weights = {}
        for rank in in_neighbors:
            last_weight_time = bf.win_update(name=self.name_time,
                    self_weight=0., neighbor_weights={rank: 1})
            event_delta = self.event_number - last_weight_time[1]
            weight_time_delta = time.time() - last_weight_time[0] \
                              + (0 if event_delta < 0 else 2 * (event_delta))
            exp = math.exp(-(weight_time_delta) / self.func_a + self.func_b) + self.func_c
            dst_weights[rank] = exp if exp > 0 else 0
        #a = torch.randn(1)
        #print(a, dst_weights)
        self_weight = self.self_weight
        norm = len(dst_weights.values())
        for rank, weight in dst_weights.items():
            dst_weights.update({rank: (1 - self_weight) * weight / (norm)})
        self_weight = 1 - sum(dst_weights.values())
        #if self_weight > 0.5:
        #print(dst_weights, self_weight, event_delta, weight_time_delta, last_weight_time[0])

        for name, tensor in sorted(self.state_dict().items()):
            bf.win_update(name=name,
                self_weight=self_weight, neighbor_weights=dst_weights)
            self.state_dict()[name].data[:] = tensor


class NodeNetwork:
    def __init__(self, params):
        self.name_stop_train = params.tensor_stop_train_name
        self.stop_train = torch.FloatTensor([-1.])
        self.ready_nodes_frac = params.accum_need_ready_neighs_frac

        self.model = Net(params)
        self.optimizer = optim.SGD(self.model.parameters(), lr=params.train_lr * bf.size())
        bf.broadcast_parameters(self.model.state_dict(), root_rank=0)
        bf.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.model.win_create_weights()

        self.topo = Topo(params)

    def _metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = bf.allreduce(tensor, name=name)
        return avg_tensor.item()

    def _prepare_to_train(self):
        bf.barrier()
        self.model.event_number = 0

        self.stop_train = torch.FloatTensor([0.])
        bf.win_create(self.stop_train, name=self.name_stop_train, zero_init=True)
        bf.win_put(self.stop_train, name=self.name_stop_train)
        #print("EPOCH", self.stop_train)

    def _prepare_to_test(self):
        self.stop_train = torch.FloatTensor([1.])
        bf.win_put(self.stop_train, name=self.name_stop_train)
        #print(self.stop_train)

    def wait_actual_neigh_weights(self, to_neighbors, from_neighbors, name_time):
        actual_node_needed_count = math.floor(
            self.ready_nodes_frac * (1 + len(from_neighbors)))
        wait_nodes = True
        while wait_nodes:
            ready_nodes_count = 0
            for rank in from_neighbors:
                last_weight_time = bf.win_update(name=name_time,
                    self_weight=0., neighbor_weights={rank: 1})
                #if bf.rank() == 0:
                #        print(self.model.event_number, last_weight_time[1])
                if self.model.event_number <= last_weight_time[1]:
                    ready_nodes_count += 1
            #print(f"ready_nodes_count [{bf.rank()}]: {ready_nodes_count}/{actual_node_needed_count}")
            #if ready_nodes_count > 0:
            #    print(f"actual_node_needed_count: {bf.rank()} {actual_node_needed_count} {ready_nodes_count} {from_neighbors}")
            #if bf.rank() == 0:
            #    print("ready", ready_nodes_count, actual_node_needed_count)
            if ready_nodes_count >= actual_node_needed_count:
                wait_nodes = False

    def train(self, dataset):
        self._prepare_to_train()

        self.model.train()
        dataset.train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            self.stop_train = bf.win_update(name=self.name_stop_train, reset=True)

            if bf.rank() == 0:
                #print(self.stop_train)
                time.sleep(0.3)
            if self.stop_train > 0.0001:
                break

            to_neighbors, from_neighbors = self.topo.get_next_neights()
            #print(f"topo [ev {model.event_number}][{bf.rank()}]: in {from_neighbors}")
            self.wait_actual_neigh_weights(to_neighbors, from_neighbors, self.model.name_time)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.model.win_put_weights()
            self.model.win_update_weights(from_neighbors)

            if batch_idx % params.log_interval == 0:
                print("[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        bf.rank(),
                        epoch,
                        batch_idx * len(data),
                        len(dataset.train_sampler),
                        100.0 * batch_idx / len(dataset.train_loader),
                        loss.item()))

    def test(self, dataset, record):
        self._prepare_to_test()

        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        for data, target in dataset.test_loader:
            output = self.model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)
                                    ).cpu().float().sum().item()

        test_loss /= len(dataset.test_sampler) if dataset.test_sampler else len(dataset.test_dataset)
        test_accuracy /= len(dataset.test_sampler) if dataset.test_sampler else len(dataset.test_dataset)

        # Bluefog: average metric values across workers.
        test_loss = self._metric_average(test_loss, "avg_loss")
        test_accuracy = self._metric_average(test_accuracy, "avg_accuracy")

        if bf.rank() == 0:
            print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                test_loss, 100.0 * test_accuracy), flush=True)
        record.append((test_loss, 100.0 * test_accuracy))
        bf.win_free(name=self.name_stop_train)

    def finish(self):
        self.model.win_put_weights()
        self.model.win_update_weights(bf.in_neighbor_ranks())


if __name__=="__main__":
    bf.init()

    params = ParamLoader("config.json")
    data = Dataset(params)
    distr_network = NodeNetwork(params)

    test_record = []
    for epoch in range(1, params.train_epochs + 1):
        distr_network.train(data)
        distr_network.test(data, test_record)

    distr_network.finish()
    print(f"[{bf.rank()}]: ", test_record)

    bf.barrier()
    bf.win_free()
