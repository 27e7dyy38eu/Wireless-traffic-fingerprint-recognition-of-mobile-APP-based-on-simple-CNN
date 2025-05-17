## 基于简单CNN的手机APP无线流量指纹识别  
版本号：V1.0  
作者：Han  
邮箱：hanz78843@gmail.com  
最后更新日期:2025/05/17  
## 项目简介  
目前项目处于初级阶段，作者正在努力维护和优化(QAQ)！  
项目演示视频：https://www.bilibili.com/video/BV1dAE8zBEuo/?spm_id_from=333.337.search-card.all.click&vd_source=b25323c9eee5af5064b19d1aa77a24a6
![攻击场景](https://github.com/27e7dyy38eu/img/blob/main/attck.png)   
随着移动互联网的高速发展，手机成为了生活中最重要工具，因此针对手机终端的隐私窃取行为也层出不穷。然而少有团队对手机无线加密流量进行分析。造成这一现象的原因有：1）不同设备、不同应用的无线流量混杂，难以区分；2）无线流量多为加密流量，难以进行分析；3）无线流量数量庞大，处理复杂度高；4）现有分析技术过于复杂，难以复现。因此，本项目在前人的基础上，提出了一种简单的、可复现的攻击方法：基于简单CNN的APP无线流量指纹识别。  
目前项目已实现10种手机APP的无线流量指纹识别。通过抓取APP启时的特征流量，在小批量数据集上的识别正确率超97%
![混淆矩阵](https://github.com/27e7dyy38eu/img/blob/main/confusion_matrix.png) 

## 先验知识
- 802.11帧分类及数据帧结构
- 流量汇聚矩阵TAM
- 卷积神经网络CNN
- 一点点python

## 项目结构
Model：储存训练好的模型  
Visualization：进行可视化，可视化图像保存在这个文件夹下的logs中  
wireless_pcap：保存pcap的文件夹，里面包含多个label的文件夹，由于安全原因，我不会提供原始的pcap，只会提供处理好的训练数据TAM  
TAM：处理好的训练数据TAM，里面包含多个label的文件夹  
1_TAM_extraction.py：这个脚本能将一个文件夹下的所有pcap中提取成TAM  
2_train_fast.py&2_train_test.py：这个脚本可以训练CNN，两个脚本仅有速度上的差别  
3_test.py：这个脚本允许你通过路径访问一个TAM的csv文件，进行分类预测  

## 🎂🎂🎂如何食用本项目？🎂🎂🎂
1）下载并打开名为“Wireless-traffic-fingerprint-recognition-of-mobile-APP-based-on-simple-CNN”的文件夹  
2）安装好python环境  
3）作者在每个py文件的名字上添加了序号，请按序号运行！  
4）可视化在Visualization中，请自行查看！  

## 联系方式
有任何问题或交流学习可联系   
我的邮箱：hanz78843@gmail.com








