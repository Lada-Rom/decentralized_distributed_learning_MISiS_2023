# Децентрализованное распределённое обучение с изменяющейся топологией

## Содержание
1. [Структура репозитория](#структура-репозитория)
2. [Требования к окружению](#требования-к-окружению)
    1. [Связь узлов](#связь-узлов)
    2. [Необходимые пакеты](#необходимые-пакеты)
3. [Запуск](#запуск)
    1. [Обучение и тестирование](#обучение-и-тестирование)
    2. [Топологии](#топологии)
    3. [Аккумулирование весов](#аккумулирование-весов)
    4. [Вывод вспомогательной информации](#вывод-вспомогательной-информации)
4. [Авторы](#авторы)

## Структура репозитория

* `async_learning.py` - python-скрипт распределённого децентрализованного асинхронного обучения при помощи bluefog на примере нейронной сети для детектирования рукописных цифр (MNIST) с возможностью смены топологии;
* `config.json` - настройки обучения для python-скрипта в json-формате;
* `nodes_cp_file.sh` - bash-скрипт для рассылки файла по узлам (например, рассылка файла с изменёнными параметрами) с помощью ssh (имена узлов/ip-адреса прописываются внутри скрипта), пример использования: `nodes_cp_file.sh <path-to-file> <dir-on-nodes>`.

## Требования к окружению

Распределённое обучение на удалённых узлах реализуется при помощи функционала [Bluefog](https://github.com/Bluefog-Lib/bluefog/tree/v0.2.2). Этот пакет использует openmpi и ssh для подключения к удалённым узлам. Общая логика обучения на нескольких узлах такова: инициирующий обучение узел распределяет задачи между узлами сети и запускает расчёт на каждом узле сети при помощи ssh. При этом команда запуска скрипта обучения остаётся неизменной относительно запуска на изначальной машине, следовательно:
1. изначальный узел должен уметь подключаться к остальным узлам по ssh, используя тоько ip-адрес (без указания порта или имени пользователя), без дполонительных диалогов;
2. файл скрипта обучения и данные должны лежать на всех нодах в одном и том же месте;
3. все зависимые пакеты должны быть доступны на всех узлах во время обучения.

### Связь узлов

Для облегчения настройки связи между узлами, которые могли изначально не настраиваться под распределённое обучение, на всех машинах создаётся новый пользователь не в группе sudo:
```
sudo adduser mpiuser
```
Таким образом, для автоматического подключения по ssh не потребуется указывать пользователя. Кроме того сервис ssh должен быть настроен на одном и том же порту на всех узлах.

Если планируется обучение на узлах, которые не имеют публичного ip-адреса, но находятся не в локальной сети, может понадобиться использование сторонних сервисов для добавления всех узлов в одну сеть. Для данного проекта используется пакет [Zerotier](https://docs.zerotier.com/start/):
```
sudo snap install zerotier          # скачивание пакета
sudo zerotier-cli join <network-id> # добавление узла в созданную сеть
```
Для подключения к узлам сети по ssh во время обучения без возникновения дополнительных диалогов, требуется:
1. заранее инициировать подключение к каждому узлу хотя бы по одному разу для подтверждения подключения;
2. создать пару из приватного и публичного ключей без пароля и обменяться этими ключами с остальными узлами сети:
```
ssh-keygen -t rsa
ssh <hostname/ip-address> mkdir -p .ssh
cat ~/.ssh/id_rsa.pub | (ssh <hostname/ip-address> "cat >> ~/.ssh/authorized_keys")
ssh <hostname/ip-address>
```
В результате последняя команда должна подключить к удалённому узлу без ввода пароля и дополнительных подтверждений.

### Необходимые пакеты

В данном проекте использовался пакет `Bluefog` версии `0.2.2`. Для его функционирования требуются некоторые дополнительные компоненты:

1. `openmpi 4.0.7`: необходимо скачать архив [openmpi-4.0.7.tar.bz2](https://www.open-mpi.org/software/ompi/v4.0/), далее его можно будет загружать на другие узлы при помощи `scp`:
```
scp openmpi-4.0.7.tar.bz2 <sudo-user>@1<ip-address>:.
tar xf openmpi-4.0.7.tar.bz2
cd openmpi-4.0.7
./configure --prefix=/opt/openmpi-4.0.7
make -j $(nproc) all 2>&1
sudo mkdir /opt/openmpi-4.0.7
sudo make install 2>&1
sudo nano /etc/environment      # добавить /opt/openmpi-4.0.7/bin: в начало значения PATH
```
2. `flatbuffers 2.0.0`:
```
git clone https://github.com/google/flatbuffers
cd flatbuffers
git checkout tags/v2.0.0
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
sudo make install
```
3. `boost 1.76.0`:
```
wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
tar xvf boost_1_76_0.tar.gz
sudo apt install python3-dev autotools-dev libicu-dev libbz2-dev
cd boost_1_76_0
sudo ./bootstrap.sh --prefix=/usr/local
./b2
sudo ./b2 install
ls /usr/local/lib/libboost_*
```
4. `torch 1.4.0`, `torchvision 0.5.0`, `networkx 2.8.8`:
```
pip install torch==1.4.0 torchvision==0.5.0 networkx==2.8.8
```

Установка `bluefog 0.2.2`:
```
git clone https://github.com/Bluefog-Lib/bluefog.git
cd bluefog
git checkout tags/v0.2.2
pip install -e . --verbose
sudo nano /etc/environment      # добавить /home/mpiuser/.local/bin: в начало значения PATH
```

## Запуск

Пример запуска скрипта обучения при помощи `bluefog`:
```
bfrun -np 24 -H worker3:4,worker4:4,worker5:4,worker6:4,worker7:4,worker8:4 python3 decentralized_distributed_learning_MISiS_2023/async_learning.py
```
В приведённом примере используются имена узлов `worker<i>`, ip-адреса которых прописаны в файле `/etc/hosts`:
```
172.30.10.13 worker3
172.30.10.14 worker4
172.30.10.15 worker5
172.30.10.16 worker6
172.30.10.17 worker7
172.30.10.18 worker8
```
При этом указание общего числа процессов (`-np 24`) является обязательным, как и распределение этих процессов между узлами (`-H worker3:4,worker4:4,worker5:4,worker6:4,worker7:4,worker8:4`), при этом их сумма должна совпадать с указанным общим числом. Более подробная информация доступна в [документации Bluefog](https://bluefog-lib.github.io/bluefog/index.html).

Данные для обучения и тестирования скачиваются автоматически при первом запуске, при этом их объём зависит от общего числа процессов. Поэтому рекомендуется заранее на каждой ноде запустить скрипт с необходимым числом процессов или скопировать эти данные с одного узла на все остальные при помощи `scp`.

Параметры, влияющие на процесс обучения, заранее указываются в файле `config.json`. При изменении каких-либо параметров на одном из узлов, необходимо разослать обновлённый файл по сети узлов:
```
nodes_cp_file.sh config.json decentralized_distributed_learning_MISiS_2023/
```
Далее будут рассмотрены параметры, доступные в конфиге. Слова "узел" и "нода" с точки зрения скрипта обучения имеют смысл виртуальныех процессов на физических узлах, а не сами физические узлы (на одном физическом узле запущены несколько процессов). Для удобства, параметры алгоритма разбиты по тематике.

### Обучение и тестирование

* `train_batch_size` - размер батча для обучения;
* `test_batch_size` - размер батча для тестирования;
* `train_lr` - темп обучения (learning rate);
* `train_epochs` - количество эпох обучения.

### Топологии

Подробнее о топологиях в `bluefog` можно узнать в их [документации](https://bluefog-lib.github.io/bluefog/topo_api.html).

* `topo_enable_dynamic` - флаг, влияющий на то, будет ли использоваться динамическая топология (смена входящих и исходящих узлов на каждой итерации обучения);
* `topo_static_kind` - код статической топологии используется, если флаг динамической топологии имеет значение `false`:
  - `0` - `FullyConnectedGraph`,
  - `1` - `RingGraph` - имеет параметр направления `topo_static_ring_direction`,
  - `2` - `StarGraph` - имеет параметр центральной ноды `topo_static_star_center`,
  - `3` - `MeshGrid2DGraph` - имеет параметр формы (размера) сети `topo_static_meshgrid_size`,
  - `4` - `SymmetricExponentialGraph` - имеет параметр "основания степени" (см. документацию) `topo_static_exp_base`,
  - `5` - `ExponentialGraph` - имеет параметр "основания степени" (см. документацию) `topo_static_exp_base`;
* `topo_static_ring_direction` - направление для кольцевой топологии:
  - `0` - двунаправленная топология (по умолчанию),
  - `1` - направление влево,
  - `2` - направление вправо; 
* `topo_static_star_center` - номер ноды (процесса), которая будет центром топологии "звезда", по умолчанию `0`;
* `topo_static_meshgrid_size` - форма/размер топологии `meshgrid`: `[0, 0]` по умолчанию - будет выбрана наиболее близкая к квадрату форма, например для 24 процессов - `[4, 6]`, можно указать необходимую форму, но она должна соответствовать общему числу процессов при запуске оубчения.
* `topo_static_exp_base` - параметр экспоненциальных топологий, в которыхх узлы соединяются только с теми узлами, разность номеров с которыми равна некоторой степени этого параметра, по умолчанию 4 для `SymmetricExponentialGraph` и 2 для `ExponentialGraph`.

### Аккумулирование весов

* `accum_need_ready_neighs_frac` - необходимая доля "готовых" соседей (тех, которые прислали свои веса для итерации не ниже итерации текущего узла) для продолжения обучения на текущем узле;
* `accum_func_params` - `[a, b, c]` - численные параметры функции агрегирования весов текущего узла и результатов расчёта соседей:  $f(t) = exp(-t/a + b) + c$, где $t$ - разность времени расчёта весов для текущей итерации на текущем узле и итерации, расчитанной на соседнем узле;
* `accum_initial_self_weight` - изначальная доля, с которой будут браться веса модели, расчитанные на текущей итерации текущим узлом, в последствии она может увеличиться (если веса соседей слишком устарели) или незначительно уменьшиться (веса соседей достаточно актуальны, происходит нормализация долей, чтобы их сумма была равна 1):

<p align="center">
$f_s + f_{neigh1} + ... + f_{neighN} = 1$
</p>
<p align="center">
$weight_{curr} = f_s \cdot weight_s + f_{neigh1} \cdot weight_{neigh1} + ... + f_{neighN} \cdot weight_{neighN}$
</p>

### Вывод вспомогательной информации

* `log_interval` - частота вывода (относительно итераций) информации об обучении;
* `log_topo` - нужен ли вывод дополнительной информации о топологии;
* `log_weights_accum` - нужен ли вывод информации о коэффициентах, с которыми агрегируются веса модели;
* `log_ready_nodes` - нужен ли вывод о готовности (для продолжения обучения) узлов;
* `log_early_stop` - нужен ли вывод о раннем прекращении эпохи, по её завершении какими-то узлами.

## Авторы

  - *Клейменов Артём*
  - *Толстенко Лада*
