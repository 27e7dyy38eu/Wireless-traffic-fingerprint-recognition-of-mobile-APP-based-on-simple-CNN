# 基于简单CNN的手机APP无线流量指纹识别

> 通过天线抓取空口流量，得知被攻击者正在或已经使用了什么手机APP

## 项目简介  
项目演示：https://www.bilibili.com/video/BV1dAE8zBEuo/?spm_id_from=333.337.search-card.all.click&vd_source=b25323c9eee5af5064b19d1aa77a24a6
![攻击场景](https://github.com/27e7dyy38eu/img/blob/main/attck.png)   
本项目扮演上图攻击场景中的攻击者  
通过天线抓取空口的802.11帧流量，进行特征提取，使用卷积神经网络模型进行APP分类，进而识别被攻击者在手机上使用了什么APP  
目前项目仅实现了10种APP的无线流量分类  
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








