# WHUME
## 魔眼(magic eyes)
IRWHU-魔眼识图SDK主要由X个代码模块组成，包括：
* irwhume (**核心模块**):
    * deepnet (**深度神经网络模型**)， 定义了项目运用到的所有深度学习网络架构类。
    * skills (**魔眼技能模块**)，定义了魔眼识图实现的所有技能类，每一个技能类即是一项功能，现有技能包括：
        * IdCardReader (**身份证识别**)
        * DriverLicenseReader (**驾驶证识别**)
        * BankCardReader (**银行卡识别**)
        * FaceMatcher (**人脸匹配**)
        * BrokenScreenInspector (**碎屏检测**)
    * scanner (**扫描器**)，定义了魔眼识图扫描图像的类对象，支持多种图像区域扫描方式，包括：
        * SliderWindow，使用滑动窗口遍历图像的每个小区块。
        * FaceScanner (**人脸定位**)，从图像中定位可能的人脸区域。
    * commons (**通用函数库**)，定义了魔眼用到的各种通用函数。
    * preprocess (**预处理**)，包括一系列预处理脚本，包括训练数据生成，模型训练，模型准确率交叉检验等。
* 3dpart (**第三方模块**):
    * [opencv](http://opencv.org/)，基于BSD许可（开源）的图像处理软件包，包含图像处理的诸多通用算法。
    * [caffe](http://caffe.berkeleyvision.org/)，基于BSD-2许可（开源）的深度卷积神经网络软件包,方便架构和编写自定义深度神经网络模型。
* dataset (**训练和测试数据集**)
