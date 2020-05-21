### 说明
```
对应 http://172.16.105.173:8899/Hao.zhang/visionproject/-/tree/tensorRT/framework/tensorrt_framework/tensorrtFrmame 目录
已经测试通过的环境：172.16.104.199
已经测试通过的代码，参考nvidia@172.16.104.199:/srv/VisionProject
```

## 1.install TensorRT-6.0
```
https://github.com/NVIDIA/TensorRT/tree/release/6.0
相关bug修复：
共有三个文件需要在编译前修改:
TensorRT/include/NvInferPluginUtils.h
TensorRT/plugin/priorBoxPlugin/priorBoxPlugin.cpp
TensorRT/parsers/caffe/caffeParser/opParsers/parsePReLU.cpp
修改后的三个文件参考 tensorrt6.0_fixbugs 文件夹
编译完成后，在 build/out 目录下libnvcaffeparser.so*，libnvinfer_plugin.so* 为本代码需要。建立对应的软连接。
已经编译通过的代码，参考nvidia@172.16.104.199:/home/nvidia/TensorRT
```

## 2.install
see [/framework/tensorrt_framework/README.md](http://172.16.105.173:8899/Hao.zhang/visionproject/-/blob/tensorRT/framework/tensorrt_framework/README.md) for details

## 3.major changes
```
1.删除原始所有自定义Plugin，替换为[PluginV2](https://github.com/NVIDIA/TensorRT/tree/release/6.0/plugin)，以及已经在tensorrt6支持的层
2.由于使用了新的reshape ，对应caffe prototxt 中 reshape 层需要修改。
3.性别年龄模型中使用的prelu已经被支持，修复bug后，对应原始 preluPlugin不再需要
4.3D关键点模型中使用的global pooling已经被pooling支持，对应原始 avgchanPlugin不再需要
5.TensorRT/parsers/caffe/caffeParser/opParsers/parsePReLU.cpp 的修改对应为Tensorrt6 的bug，此bug在Tensorrt7下已经修复。Tensorrt6需要修改对应文件。
参考：https://github.com/NVIDIA/TensorRT/issues/179
https://github.com/yqbeyond/TensorRT/pull/1/files
6.TensorRT/include/NvInferPluginUtils.h TensorRT/plugin/priorBoxPlugin/priorBoxPlugin.cpp 对应修复bug: 无法加载PriorBox层的prior_box_param 的参数min_size，max_size，aspect_ratio。
7.删除自定义Plugin后，不再需要的代码被移除，对应onnx解析相关代码被移除，只保留caffe解析器。
```

## 4. models
```
原始的caffemodel未改变，对应的prototxt需要调整。
调整后的模型参考：[Visionproject_models](http://172.16.105.173:8899/Hao.zhang/visionproject_models),此目录下的所有模型已经被测试无误。
注意：
为了保持ssd模型的命名区分，ssd模型的命名需要以SSD开头，原始的headFace360_640MeanVariance 被重新命名为SSD_HF_360x640，当前只加载一次prototxt文件，_ori.prototxt不再需要。
对应的，需要在 [tensorrt_loader.py](http://172.16.105.173:8899/Hao.zhang/visionproject/-/blob/tensorRT/framework/loader/tensorrt_loader.py) 中修改det_model_map。
```
