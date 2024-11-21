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

