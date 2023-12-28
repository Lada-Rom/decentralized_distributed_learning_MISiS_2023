from __future__ import print_function

from bluefog.common import topology_util
import bluefog.torch as bf
import os
import sys
import warnings
import time
import math
warnings.simplefilter('ignore')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.name_time = "time"
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
            last_weight_time = bf.win_update(name=self.name_time, self_weight=0.,
                neighbor_weights={r: (0 if r != rank else 1) for r in in_neighbors})
            event_delta = self.event_number - last_weight_time[1]
            weight_time_delta = time.time() - last_weight_time[0] \
                              + (0 if event_delta < 0 else 2 * (event_delta))
            exp = math.exp(-(weight_time_delta) / 10.0 + 0.98) - 1.7
            dst_weights[rank] = exp if exp > 0 else 0
        #a = torch.randn(1)
        #print(a, dst_weights)
        self_weight = 0.5
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


train_dataset = None
train_sampler = None
train_loader = None

test_dataset = None
test_sampler = None
test_loader = None

FLAG = torch.FloatTensor([-1.])


def train(model, epoch, dynamic_neighbors, log_interval, name_flag):
    global FLAG
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):

        FLAG = bf.win_update(name=name_flag, reset=True)
        if bf.rank() == 0:
        #    print(FLAG)
            time.sleep(2)
        if FLAG > 0.0001:
            break

        to_neighbors, from_neighbors = next(dynamic_neighbors)
        #dst_weights = {rank: 1. / (len(from_neighbors) + 1) for rank in from_neighbors}
        #self_weight = 1. / (len(from_neighbors) + 1)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        model.win_put_weights()
        model.win_update_weights(from_neighbors)

        if batch_idx % log_interval == 0:
            print("[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    bf.rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = bf.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, record):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for data, target in test_loader:
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)
                                 ).cpu().float().sum().item()

    test_loss /= len(test_sampler) if test_sampler else len(test_dataset)
    test_accuracy /= len(test_sampler) if test_sampler else len(test_dataset)

    # Bluefog: average metric values across workers.
    test_loss = metric_average(test_loss, "avg_loss")
    test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    if bf.rank() == 0:
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            test_loss, 100.0 * test_accuracy), flush=True)
    record.append((test_loss, 100.0 * test_accuracy))


if __name__=="__main__":
    bf.init()
    
    batch_size = 64
    test_batch_size = 1000
    lr = 0.001
    epochs = 10
    log_interval = 10

    data_folder_loc = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")

    # train dataset
    train_dataset = datasets.MNIST(
        os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=bf.size(), rank=bf.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **{}
    )

    # test dataset
    test_dataset = datasets.MNIST(
        os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    test_sampler = None
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=bf.size(), rank=bf.rank()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, sampler=test_sampler, **{}
    )

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())

    # Bluefog: broadcast parameters & optimizer state.
    bf.broadcast_parameters(model.state_dict(), root_rank=0)
    bf.broadcast_optimizer_state(optimizer, root_rank=0)

    dynamic_neighbors = topology_util.GetDynamicSendRecvRanks(
        bf.load_topology(), bf.rank())
    
    name_flag = "end"
    model.win_create_weights()

    test_record = []
    for epoch in range(1, epochs + 1):
        bf.barrier()
        model.event_number = 0

        FLAG = torch.FloatTensor([0.])
        bf.win_create(FLAG, name=name_flag, zero_init=True)
        bf.win_put(FLAG, name=name_flag)
        #print("EPOCH", FLAG)

        train(model, epoch, dynamic_neighbors, log_interval, name_flag)

        FLAG = torch.FloatTensor([1.])
        bf.win_put(FLAG, name=name_flag)
        #print(FLAG)

        test(model, test_record)
        bf.win_free(name=name_flag)

    model.win_put_weights()
    model.win_update_weights(bf.in_neighbor_ranks())

    print(f"[{bf.rank()}]: ", test_record)

    bf.barrier()
    bf.win_free()
