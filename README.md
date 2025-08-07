# ML Model Profiling & Optimization

🚀 This repository explores the profiling and optimization of deep learning models using tools such as **PyTorch Profiler**, **NVIDIA Nsight Systems**, **ONNX**, and **TensorRT**.

> ⚠️ **Disclaimer**: This repository is strictly for **educational and interview preparation** purposes. It is not intended for deployment or production use.

---

## 📌 Purpose

The goal is to understand how deep learning models behave under various system constraints and improve their performance by:

- Profiling training and inference pipelines  
- Identifying and fixing bottlenecks in data loading, GPU utilization, and memory usage  
- Experimenting with inference optimization using TorchScript, ONNX, and TensorRT  
- Analyzing GPU kernel launches using Nsight Systems  
- Understanding real-world deployment constraints (batch size, latency, throughput)  

---

## 🧰 Tools & Frameworks

- PyTorch Profiler  
- NVIDIA Nsight Systems  
- ONNX  
- TensorRT  
- Python Profilers: `cProfile`, `torch.utils.bottleneck`, `line_profiler`  

---

## 📁 Folder Structure

- `resnet_profiling/` – PyTorch model profiling scripts  
- `dataloader_benchmarking/` – DataLoader performance tests  
- `onnx_tensorrt/` – Inference optimization experiments  
- `nsight_traces/` – Nsight command scripts and trace outputs  
- `utils/` – Helper scripts and profiling wrappers  
- `README.md` – This file  

---

## 🧪 Key Experiments

- ✅ Compare `num_workers` in DataLoader and its effect on throughput  
- ✅ Profile model layers for CUDA and CPU time  
- ✅ Export model to ONNX and run with TensorRT (FP32 vs FP16)  
- ✅ Visualize kernel execution with Nsight Systems  
- ✅ Document top 5 bottlenecks and propose mitigation strategies  

---

## 🚀 How to Run

### 1. Set up environment (Recommended: Conda)

Run the following commands in your terminal:

- `conda create -n mlopt python=3.10 -y`  
- `conda activate mlopt`

### 2. Install dependencies

Use the following pip commands:

- `pip install -r requirements.txt`

---

### 3. Run Experiments

|                      Step                           |                                            Command                                                  |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Export model to ONNX                                | `python export_onnx.py`                                                                             |
| Build TensorRT engine (ensure `trtexec` is in PATH) | `trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.trt --fp16`                                |
| Profile model with PyTorch                          | `python profiler_resnet.py`                                                                         |
| Benchmark DataLoader performance                    | `python data_loader_test.py`                                                                        |
| Compare FP16 vs FP32 outputs                        | `python low_precision_test.py`                                                                      |
| Run modular inference pipeline                      | `python modular_pipeline.py`                                                                        |
| Line-by-line profiling                              | `kernprof -l line_profile_example.py`, then `python -m line_profiler line_profile_example.py.lprof` |

---

## 📄 License

This repository is released under the **MIT License**, but again — it is intended **only for educational learning and technical exploration**.

---

## 🙋‍♂️ About

Maintained by [Pranav Venkatesan](https://www.linkedin.com/in/pranav31/).  
Feel free to fork, star, or reach out for feedback!
