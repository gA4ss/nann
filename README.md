# 几年前重新捡起神经网络时编写的反馈神经网络

## pynann编译与安装

在**nann**目录中使用以下命令进行安装
```
python setup.py build
sudo python setup.py install
```
安装完成后，在_python_代码中使用`import pynann`，即可使用。

## 环境设定
首先配置环境变量`NANN_HOME`指向某个目录，此目录是	工作目录例如`~/.nann`中。
随后在`~/.nann`中建立`lib`,`etc`,'alg','log'。四个目录。分别用来存放扩展库，
配置文件，算法扩展以及日志文件。

## 配置文件
在`etc`目录中，放置以下两种配置文件。_ann_nannmgr.json_，是_**nann**_的配置文件，
_ann_alg_buildin.json_是真对某项算法默认配置。

### _ann_nannmgr.json_
```json
{
        "enable_log": false
}
```

### _ann_alg_buildin.json_
{
        "eta": 0.05,
        "momentum": 0.03,
        "threshold": 0,
        "alg": "logistic"
}

## Python使用说明
```python
#!/usr/bin/env python
import os
import pynann

task = "HELLO"
ann_json = "/Users/devilogic/Naga/nann/nann_dev/doc/create.json"
input_json = "/Users/devilogic/Naga/nann/nann_dev/doc/input.json"

"""开始训练"""
pynann.training(task, ann_json, input_json, is_file=True)

"""等待训练完成"""
pynann.wait(task)

"""测试是否工作完成"""
print pynann.done(task)

"""设置输出精度"""
pynann.set_precision(2)

"""获取map结果"""
map_results = pynann.get_map_results(task)
print map_results

"""获取reduce结果"""
reduce_result = pynann.get_reduce_result(task)
print reduce_result

"""等待所有任务结束"""
pynann.waits()

"""清除内存"""
pynann.clears()

```

## 错误代码

## json输入详解

### create.json
```json
{
	"ann": {
		"alg": "ann_alg_logistic",
		"weight matrixes": {
			"0": {
				"r1": [0.75, 0.83, 0.39],
				"r2": [0.98, 0.43, 0.12],
				"r3": [0.12, 0.45, 0.78],
				"r4": [0.11, 0.12, 0.65],
				"r5": [0.56, 0.67, 0.34]
			},
			"1": {
				"r1": [0.35, 0.23, 0.19, 0.37],
				"r2": [0.51, 0.49, 0.72, 0.11],
				"r3": [0.24, 0.31, 0.71, 0.51]
			},
			"2": {
				"r1": [0.15, 0.25, 0.41],
				"r2": [0.21, 0.29, 0.25],
				"r3": [0.34, 0.11, 0.87],
				"r4": [0.61, 0.92, 0.93]
			}
		},
		"delta weight matrixes": {
			"0": {
				"r1": [0.07, 0.08, 0.09],
				"r2": [0.09, 0.04, 0.02],
				"r3": [0.01, 0.04, 0.08],
				"r4": [0.01, 0.02, 0.05],
				"r5": [0.05, 0.07, 0.04]
			},
			"1": {
				"r1": [0.05, 0.03, 0.09, 0.07],
				"r2": [0.01, 0.09, 0.02, 0.01],
				"r3": [0.04, 0.01, 0.01, 0.01]
			},
			"2": {
				"r1": [0.05, 0.05, 0.01],
				"r2": [0.01, 0.09, 0.05],
				"r3": [0.04, 0.01, 0.07],
				"r4": [0.01, 0.02, 0.03]
			}
		}
	}
}
```
### input.json
```json
{
	"samples": {
		"t1": {
			"input": [0.1, 0.2, 0.34, 0.45, 0.55],
			"target": [0.11, 0.23, 0.78]
		},
		"t2": {
			"input": [0.21, 0.6, 0.4, 0.23, 0.51],
			"target": [0.11, 0.23, 0.78]
		},
		"t3": {
			"input": [0.31, 0.7, 0.31, 0.1, 0.46],
			"target": [0.11, 0.23, 0.78]
		},
		"t4": {
			"input": [0.41, 0.9, 0.21, 0.61, 0.78],
			"target": [0.11, 0.23, 0.78]
		}	
	}
}
```
