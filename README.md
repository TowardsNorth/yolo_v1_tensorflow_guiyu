## YOLO_tensorflow

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.

### Installation


1. Download Pascal VOC dataset, and create correct directories
	```Shell
	$ ./download_data.sh
	```

2. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `data/weight`

3. Modify configuration in `yolo/config.py`

4 论文解读： Training:  先构建一个用于分类的网络，在ImageNet 1000-class上进行预训练。用这个网络训练一周，然后在ImageNet上的top-5准确率达到88%。                        然后将这个分类的网络进行改造，进行检测训练。
	
            Inference:  end to end, 直接从网络中得出预测值
	  
5  源码解读： Taining:  训练阶段网络的输出为[batch_size, cell_size, cell_size, C+5*B], 得到的x_center,y_center是相对于该cell左上角的偏移值，                  width, height是相对于整张特征图的比率。注意label与网络输出的一一对应。另外，损失函数那部分是重点

             Inference: 注意与训练阶段网络的输出一一对应即可


             其它具体的细节可以看源码注释，我对于每一个细节都做了源码注释。
	    
6  效果显示： 运行已经训练好的模型文件（第2步下载得到，并且放到相应的文件夹下），测试结果如下
            
	    ![yolov1](yolo_v1_tensorflow_guiyu/test/yolov1.JPG)



### Requirements
1. Tensorflow

2. OpenCV
