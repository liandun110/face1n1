# 代码结构
```bash
face1n1/
  images/
  logs/
  asserts/
    whl/
      *.whl
  tools/
    env_config.sh
    det_10g.onnx
    20241112_024744_ep02.onnx  # https://huggingface.co/liandun110/face_recognition/resolve/main/20241112_024744_ep02.onnx  
  pyfat_implement.py
  README.md
```

# 使用说明
```bash
conda create -n face1n1 python=3.10.12
conda activate face1n1
cd tools/
bash env_config.sh
cd ..
python pyfat_implement.py
tar -cvf 1n1_ubuntu_cu12_<版本号>.tar ./
```

# 测评流程
1. 打为 tar 包
2. 将 tar 包拷贝到公安网U盘，密码是 `111111`。
3. 将公安网U盘中的tar包拷贝到公安网机器上
4. 利用 filezilla 软件将公安网机器上的tar包拷贝到云平台上：Host=11.33.4.180 Username=myftp Password=myftp
5. 在浏览器登录视重通测试平台，网址：`11.33.4.180:8083/login`，账号为 `sys`，密码为 `111111`。
6. 在浏览器登录 11.33.4.180:8083/taskList，单击`新建任务`，输入tar包的完整名称即可。

# 测评包记录

|  测评包名称   | 测评包说明 |
|:--------:| :-------: |
| 11290949 | 1. 解决了np_gallery[~np.array(is_usable_list), :] = 0这句话在平台上的报错. 2. 模型为：0913_backbone_ep0000.onnx |
| 11290951 | 1. 解决了np_gallery[~np.array(is_usable_list), :] = 0这句话在平台上的报错. 2. 模型为：20241119_003521_ep19.onnx |

# 2024视重通厂商得分

**人脸**：共 66 个结果
| 排名 | 得分    |
| :----: | :-------: |
| 1   | 0.9215% |
| 2   | 0.9215% |
| 3   | 0.9460% |
| 4   | 0.9505% |
| 5   | 0.9603% |
| 6   | 0.9746% |
| 7   | 0.9978% |
| 8   | 1.0266% |
| 9   | 1.0407% |
| 10   | 1.0423% |
| 11   | 1.1122% |
| 12   | 1.1403% |
| 13   | 1.1536% |
| 14   | 1.2163% |
| 15   | 1.2198% |
| 16   | 1.2250% |
| 17   | 1.3186% |
| 18   | 1.3394% |
| 19   | 1.3932% |
| 20   | 1.4003% |
| 21   | 1.4164% |
| 22   | 1.4597% |
| 23   | 1.4674% |
| 24   | 1.4727% |
| 39   | 1.9834% |
| 40   | 2.0126% |
| 44   | 2.0545% |
| 45   | 2.0807% |
| 49   | 2.9258% |
| 50   | 3.1363% |
| 53   | 3.8303% |
| 54   | 4.6065% |
| 60   | 6.6125% |
| 61   | 10.4991% |
| 66   | 80.4628% |

**车牌**：共36个结果
| 排名 | 得分    |
| :----: | :-------: |
| 1   | 0.8666% |
| 2   | 0.8748% |
| 3   | 0.8899% |
| 4   | 0.9267% |
| 5   | 0.9719% |
| 6   | 0.9754% |
| 7   | 1.3290% |
| 8   | 1.3297% |
| 35   | 9.7101% |
| 36   | 11.7405% |

# 横幅识别检测项目
1. https://github.com/ArkadyBIG/BannerDetection.git
2. https://github.com/iboraham/football-banner-detection-model.git
3. https://github.com/bilawalHussain5646/Banner-Recognition-System.git

