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

