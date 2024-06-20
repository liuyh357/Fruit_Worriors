### 最终模型权重链接：https://pan.quark.cn/s/1c040ba17514

### 项目根目录

│── 团队大作业报告.pdf
│── 运行截图/
│── frontend/
│── label_data/
│── predict.py
│── test/
│── xml2txt.py
└── README.MD
└── requirements.txt


1. 团队大作业报告.pdf
   包含项目的详细描述、方法、结果以及分析。
3. 运行截图
   包含项目运行过程中生成的截图，用于展示模型的预测效果。
4. frontend
   包含前端代码，用于展示和交互模型预测结果的用户界面。
5. label_data
   标注的20张图片。
6. predict.py
   用于执行模型预测的脚本。运行此脚本可以输出所需的16个参数（实际上是17个，我们额外输出多了一个mix_iou）。
7. test
   包含用于predict.py预测的60张图片和`XML`文件，用于验证模型的性能。
8. xml2txt.py
   用于将标注数据从`XML`格式转换为`TXT`格式的脚本，以符合yolo模型训练/预测的格式。
9. README.MD
   项目README文件，包含项目概述、目录结构、依赖项和运行说明等。
10. requirements.txt
    包含项目所需的依赖项列表，用于安装依赖项。

#### 贡献者：陈伟宁，梁斯哲，刘予桁，吴非泽，张均逸
