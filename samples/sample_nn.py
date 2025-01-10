#!/usr/bin/python
# -*- coding: utf-8 -*-

from logging import getLogger, StreamHandler, Formatter, INFO
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchviz import make_dot
from ofplang.definitions import Definitions
from ofplang.protocol import Protocol
from ofplang.validate import check_definitions, check_protocol
from ofplang.runner import Runner
from ofplang.executors import SimulatorBase
import os

# Logging setup
handler = StreamHandler()
handler.setLevel(INFO)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

getLogger('executors').addHandler(handler)
getLogger('executors').setLevel(INFO)

# PyTorch model definitions
class NNBase(nn.Module):
    def __init__(self, input_ids=None, output_ids=None):
        super(NNBase, self).__init__()
        self.input_ids = input_ids or []
        self.output_ids = output_ids or []

    def add_input(self, input_id):
        """Add an input ID to the model."""
        self.input_ids.append(input_id)

    def add_output(self, output_id):
        """Add an output ID to the model."""
        self.output_ids.append(output_id)

    def initialize_weights(self):
        """Initialize weights using a standard method."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

hidden_layer1 = 100
hidden_layer2 = 100
hidden_layer3 = 100
hidden_layer_bridge = 100

# NN1 for Red channel
class NN1r(NNBase):
    def __init__(self):
        super(NN1r, self).__init__()
        self.fc1 = nn.Linear(3, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer_bridge)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# NN1 for Green channel
class NN1g(NNBase):
    def __init__(self):
        super(NN1g, self).__init__()
        self.fc1 = nn.Linear(3, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer_bridge)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# NN1 for Blue channel
class NN1b(NNBase):
    def __init__(self):
        super(NN1b, self).__init__()
        self.fc1 = nn.Linear(3, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer_bridge)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class NN2(NNBase):
    def __init__(self):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_bridge*2, hidden_layer2*4) 
        self.fc2 = nn.Linear(hidden_layer2*4, hidden_layer_bridge)
    
    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)  # 2つのベクトルを結合
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

# NN3 for final output
class NN3(NNBase):
    def __init__(self):
        super(NN3, self).__init__()
        self.fc1 = nn.Linear(hidden_layer_bridge, hidden_layer3)
        self.fc2 = nn.Linear(hidden_layer3, 3)  # Output size is now 3 to match the 3 colors

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    




from collections.abc import Iterable
from ofplang.runner import Runner, ExecutorBase, Experiment, OperationNotSupportedError, Model
from ofplang.protocol import EntityDescription
                      
class TrainBase(SimulatorBase):
    def __init__(self, definitions, loss_fn_dict_list, num_epochs, lr=0.001):
        self.definitions = definitions
        check_definitions(self.definitions)
        self.loss_fn_dict_list = loss_fn_dict_list
        self.num_epochs = num_epochs
        self.lr = lr
        self.submodels = {}
        self.optimizer = None

    def make_submodels(self, models_info):
        self.models_info = models_info
        
        for model_info in self.models_info:
            name = model_info["name"]
            instance = model_info["instance"]
            path = model_info.get("path", None)
            
            self.submodels[name] = instance
            if path:
                self._load_model_parameters(instance, path)
            else:
                instance.apply(self._init_weights)

    def _make_optimizer(self):
        return optim.Adam([param for model in self.submodels.values() for param in model.parameters()], lr=self.lr)

    def _load_model_parameters(self, model, file_name):
        if os.path.exists(file_name):
            try:
                print(f"Loading model parameters from {file_name}...")
                state_dict = torch.load(file_name)
                model.load_state_dict(state_dict)
                print("Model parameters loaded successfully.")
            except RuntimeError as e:
                print(f"Model loading failed due to state dict mismatch: {e}")
                print("Reinitializing model parameters...")
                model.apply(self._init_weights)
            except Exception as e:
                print(f"Unexpected error while loading model: {e}")
                print("Reinitializing model parameters...")
                model.apply(self._init_weights)
        else:
            print(f"File {file_name} does not exist. Initializing model parameters...")
            model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _load_models_parameters(self):
        for model_info in self.models_info:
            instance = model_info["instance"]
            path = model_info.get("path", None)
            if path:
                self._load_model_parameters(instance, path)
            else:
                instance.apply(self._init_weights)

    def _save_model_parameters(self, model, file_name):
        try:
            torch.save(model.state_dict(), file_name)
            print(f"Model parameters saved to {file_name}")
        except Exception as e:
            print(f"Failed to save model parameters: {e}")

    def _save_models_parameters(self):
        for model_info in self.models_info:
            instance = model_info["instance"]
            path = model_info.get("path", None)
            if path:
                self._save_model_parameters(instance, path)

    def select_data(self, experiments_data):
        return numpy.random.choice(experiments_data)
    
    def train(self, experiments_data):
        self._load_models_parameters()
        self.optimizer = self._make_optimizer()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            selected_data = self.select_data(experiments_data)

            check_protocol(selected_data["protocol"], self.definitions)

            selected_data["protocol"].dump()

            runner = Runner(selected_data["protocol"], self.definitions, executor=self)

            experiment = runner.run(inputs=selected_data["input"])

            total_loss = 0.0
            for loss_entry in self.loss_fn_dict_list:
                for key, loss_fn in loss_entry.items():
                    predicted_output = eval(f"experiment.output{key}").view(-1)
                    target_output = eval(f"selected_data['output']{key}").view(-1)

                    loss = loss_fn(predicted_output, target_output)
                    total_loss += loss

            total_loss.backward()
            self.optimizer.step()

            for loss_entry in self.loss_fn_dict_list:
                for key, loss_fn in loss_entry.items():
                    predicted_output = eval(f"experiment.output{key}").view(-1)
                    target_output = eval(f"selected_data['output']{key}").view(-1)
                    print(f"predicted_output: {predicted_output}, target_output: {target_output}")

            print(f"Epoch {epoch + 1}, Loss: {total_loss.item()}")

        self._save_models_parameters()
        print("Training complete.")


class MyExecutor(TrainBase):
    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        # logger.info(f"execute: {(operation, inputs)}")
        print(f"Execute: {operation} <- {inputs}")

        outputs = {}

        """
        メモ: 
        - この関数は、それぞれのプロトコルの節点ごとに呼び出される
        - inputとoutputはここに渡される前に整形されている
        - 基本的には、operation_idごとに処理を区別することはできないが、それはそもそも区別する必要がないので、そういうtypeの命名方法になっている。
        """

        if operation.type in self.submodels:
            model_instance = self.submodels[operation.type]
            if operation.type in {"NN1r", "NN1g", "NN1b"}:
                value = model_instance(inputs["in1"]["value"])
                outputs["out1"] = {"value": value, "type": "Array[Float]"}
            elif operation.type == "NN2":
                value = model_instance(inputs["in1"]["value"], inputs["in2"]["value"])
                outputs["out1"] = {"value": value, "type": "Array[Float]"}
            elif operation.type == "NN3":
                value = model_instance(inputs["in1"]["value"])
                outputs["out1"] = {"value": value, "type": "Array[Float]"}
            else:
                raise NotImplementedError(f"Unknown operation type: {operation.type}")
        else:
            raise NotImplementedError(f"Operation type {operation.type} is not registered in submodels.")

        # モデルの出力ポートに対応した出力を生成
        _operation = model.get_by_id(operation.id)
        outputs = {port.id: {"value": value, "type": port.type} for _, port in _operation.output()}

        return outputs

# 使用例

"""
モデル情報の三つ組リストを作成
{"name": name, "instance": instance, "path": path}
"""
models_info = [
    {"name": "NN1r", "instance": NN1r(), "path": "/home/ozatamago/program/nn1r.pth"},
    {"name": "NN1g", "instance": NN1g(), "path": "/home/ozatamago/program/nn1g.pth"},
    {"name": "NN1b", "instance": NN1b(), "path": "/home/ozatamago/program/nn1b.pth"},
    {"name": "NN2", "instance": NN2(), "path": "/home/ozatamago/program/nn2.pth"},
    {"name": "NN3", "instance": NN3(), "path": "/home/ozatamago/program/nn3.pth"},
]

"""
私たちは実験データとして、次の三つ組を複数持っているとする。
{"protocol": protocol, "input": input, "output": output}
"""
experiments_data = [
    {
        "protocol": Protocol("./sample_nn.yaml"),
        "input": {
            "channel_r": {"value": numpy.array([[5.0, 0.0, 0.0]]), "type": "Array[Float]"},
            "channel_g": {"value": numpy.array([[0.0, -5.0, 0.0]]), "type": "Array[Float]"},
            "channel_b": {"value": numpy.array([[0.0, 0.0, 0.0]]), "type": "Array[Float]"}
        },
        "output": {
            "mixed_output": {"value": torch.sigmoid(torch.tensor([5.0, -5.0, 0.0])), "type": "Array[Float]"}
        }
    },
    {
        "protocol": Protocol("./sample_nn.yaml"),
        "input": {
            "channel_r": {"value": numpy.array([[1.0, 0.0, 0.0]]), "type": "Array[Float]"},
            "channel_g": {"value": numpy.array([[0.0, -1.0, 0.0]]), "type": "Array[Float]"},
            "channel_b": {"value": numpy.array([[0.0, 0.0, 0.0]]), "type": "Array[Float]"}
        },
        "output": {
            "mixed_output": {"value": torch.sigmoid(torch.tensor([1.0, -1.0, 0.0])), "type": "Array[Float]"}
        }
    }
]

loss_fn_dict_list = [{"[\"mixed_output\"][\"value\"]": nn.MSELoss()}]

train_instance = MyExecutor(
    definitions=Definitions('./manipulate.yaml'),
    loss_fn_dict_list=loss_fn_dict_list,
    num_epochs=10,
    lr=0.001
)

train_instance.make_submodels(models_info)
train_instance.train(experiments_data)



"""
train 関数の定式化 

# (outputとlossの対応づけの複数組)
loss_fn_dict_list = ["[\"mixed_output\"][\"value\"]": nn.MSELoss(), ...]

def train(experiments_data, executor=MyExecutor, models_info, loss_fn_dict_list, num_epochs, lr=0.001):
    1. submodelsの作成
    2. パラメータのロード
    3. optimizerの作成
    4. for _ in range(num_epochs):
        1. 勾配の初期化
        2. 実験データの選択
        3. プロトコルの整合性確認
        4. Runnnerのインスタンス化
        5. 実験の実行
        (6.) 出力データの整形
        7. 損失計算
        8. 勾配計算、学習
        9. ログ出力
    5. パラメータの保存

    1. 
    * submodels = make_submodels(models_info)
    submodels = {}
    for model_info in models_info:
        submodels[model_info["name"]] = model_info["instance"]
    2. 
    for model_info in models_info:
        instance = model_info["instance"]
        path = model_info.get("path", None)
        if path:
            load_model_parameters(instance, path)
        else:
            init_weights(instance)
    3. 
    optimizer = optim.Adam([param for model in submodels.values() for param in model.parameters()],lr=0.001)
    4. 
    for _ in range(num_epocs):
        1. 
        optimizer.zero_grad()
        2. 
        selected_data = select_data(experiments_data)
        3. 
        check_protocol(selected_data["protocol"], definitions)
        selected_data["protocol"].dump()
        4. 
        runner = Runner(selected_data["protocol"], definitions, executor=MyExecutor(submodels))
        5. 
        experiment = runner.run(inputs=selected_data["input"])
        6. 
        predicted_output = experiment.output["mixed_output"]["value"].view(-1)
        target_output = selected_data["output"]["mixed_output"]["value"].view(-1) 
        7. 
        loss = loss_fn(predicted_output, target_output)
        8. 
        loss.backward()
        optimizer.step()
        10. 
        print(f"predicted_output: {predicted_output}, target_output: {target_output}")
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    5. 
    for model_info in models_info:
        instance = model_info["instance"]
        path = model_info.get("path", None)
        if path:
            save_model_parameters(instance, path)

# 使い方
train(experiments_data=experiments_data, executor=Executor, models_info=models_info, (outputとlossも対応づけの複数組), num_epochs=100)

"""

"""
# 私たちは実験データとして、次の三つ組を複数持っているとする。
# ("protocol", "input": input, "output": output)

# 上記の三つ組をリストとして複数持つ
# Experiments_data = {(protocol1, input1, output1), (protocol2, input2, output2), ...}

executor = MyExecutor()
optimizer = (contain all using NNs)
loss_fn = MSE()

for i in range(100):
    selected_data = select_data(Experiments_data)
    # Runnerの設定
    runner = Runner(selected_data.protocol, definitions, executor=executor)

    # 実験の実行
    experiment = runner.run(selected_data.input)

    loss1 = loss_fn(experiment.output['some key'], selected_data.output['some key'])
    loss2 = ...
    loss3 = ...

    loss = loss1+loss2+loss3

    loss.backward()
    optimizer.step()
"""


# # 実験結果の出力
# print("Final ExperimentOutput:")
# print(experiment.output)