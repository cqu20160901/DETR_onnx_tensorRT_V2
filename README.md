# DETR_onnx_tensorRT_V2

DETR tensor去除推理过程无用辅助头+fp16部署再次加速+解决转tensorrt 输出全为0问题的新方法。

**由于模型较大无法直接上传，onnx和tensorrt [模型文件下载链接](https://github.com/cqu20160901/DETR_onnx_tensorRT_V2/releases/tag/v1.0.0)**

# 1、转tensorrt 输出全为 0 老问题回顾

&emsp;&emsp;在用 TensorRT 部署 DETR 检测模型时遇到：转tensorrt 输出全为 0 的问题。多次想放弃这个模型部署，花了很多时间查阅，最终解决方法用了两步：

&emsp;&emsp;第一步，修改onnx模型输出层Gather的参数；

&emsp;&emsp;第二步，转tensorrt 模型时不能量化，使用float32。

&emsp;&emsp;修改Gather的参数时只取最后一个输出头的结果，当时也很困惑，但没有多思考，先解决问题。后来就琢磨这个事情，既然只取最后一个头的结果，那么中间的头完全可以不要，这样就可以不使用Gather操作，且可以加快模型的推理速度。想法形成后，说干就干。**最终想法得以验证，且不会在遇到“转tensorrt 输出全为 0 问题”**。

&emsp;&emsp;转tensorrt 输出全为 0 的可能的本质原因：（1）Gather的参数中的取最后一个维度数据用的是自动推断的-1，可能是算子不支持，需改成指定的维度；（2）辅助头中数据很小超出了float16的表示范围，影响整体使用float16量化效果。

# 2、导出onnx去除辅助头，规避Gather算子

&emsp;&emsp;导出onnx去除辅助头需要修改两个地方：（1）修改TransformerDecoder，（2）修改DETR获取模型结果的代码规避Gather算子。

（1）修改TransformerDecoder

&emsp;&emsp;新增如下几行代码：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ae939511c2844139a332ad59b7849017.png)

（2）修改DETR获取模型结果的代码规避Gather算子

&emsp;&emsp;修改如下：![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d901736def7244a1823c9705610deadb.png)

修改以上两个地方后导出onnx（导出onnx后用simplify处理以下），即为没有辅助头和Gather算子的结果，效果如下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ecb5de82c6634c68974c1ce9c3de20aa.png)

# 3 模型结果验证

导出onnx的结果是否和原始的是一致的这是最让人担忧的，以下进行验证对比：

&emsp;&emsp;（1）对比原始onnx检测结果和本示例导出的onnx检测结果；

&emsp;&emsp;（2）对比原始模型导出tensonRT的速度和本示例导出的导出tensonRT的速度；

&emsp;&emsp;（3）验证使用float16量化导出的tensorRT模型结果和速度。

## （1）原始onnx检测结果和本示例导出的onnx检测结果

对比检测结果是一致的，用对比工具对比的结果也是一致的。

原始onnx的检测效果

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/fa27d09d425a4de99df7c0b1b7f6d627.jpeg)

本示例导出onnx检测结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/88c42c3582c3401590baf92b77c9e790.jpeg)

## （2）对比原始模型导出tensonRT的速度和本示例导出的导出tensonRT的速度

导出tensorRT使用的是float32没有进行量化，对同一张图像推理1000次的平均时耗，还是有轻微的加速效果。

原始模型导出tensonRT模型推理时耗（fp32）

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/34e81b54f0e74355b42fac7fd92b8f9e.png)

本示例导出的导出tensonRT模型推理时耗（fp32）

![!\[在这里插入图片描述\](https://img-blog.csdnimg.cn/direct/456bc183cf2548f59d523f6562058a7d.png](https://img-blog.csdnimg.cn/direct/fdf58955e6644fe08871c4f580c9df02.png)

# （3）验证使用float16量化导出的tensorRT模型结果和速度

基于本示例导出的onnx模型转tensorRT，对比使用float32和float16转出来的模型大小明显变小，推理速度也明显加快。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a605e69184e446a7a1cc0fcd8874208a.png)

本示例导出的导出tensonRT模型推理时耗（fp32，对同一张图像推理1000次的平均时耗）

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/01b7effbed2c43e8ae6220d4261278e2.png)

本示例导出的导出tensonRT模型推理时耗（fp16，对同一张图像推理1000次的平均时耗）

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/365e5e343bee4bec81861828ca1e27ef.png)

对比检测结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6fe291f643384a619a30dbae9364d288.png)

