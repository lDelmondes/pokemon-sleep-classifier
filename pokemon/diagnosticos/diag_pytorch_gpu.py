import torch

print(f"PyTorch versao: {torch.__version__}")
print(f"CUDA disponivel: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"CUDA compilado no torch: {torch.version.cuda}")
else:
    print("ATENCAO: PyTorch NAO esta enxergando a GPU. Vamos investigar.")