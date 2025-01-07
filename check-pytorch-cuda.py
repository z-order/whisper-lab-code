import sys
import pip
import torch

if __name__ == "__main__":
    # Python and pip versions
    print(f"Python Version: {sys.version}")
    print(f"Python Version Info: {sys.version_info}")
    print(f"Python Version Major: {sys.version_info.major}")
    print(f"Python Version Minor: {sys.version_info.minor}")
    print(f"pip Version: {pip.__version__}")

    # CUDA initialization
    try:
        torch.cuda.init()  # Try initializing CUDA explicitly
    except Exception as e:
        print("CUDA initialization error:", str(e))

    # CUDA debugging information
    try:
        print("PyTorch Version:", torch.__version__)
        print("CUDA Version:", torch.version.cuda)
        print("Is CUDA available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("Current GPU:", torch.cuda.current_device())
            print("GPU Name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA initialization error:", str(e))

    # Test Tensor operations on CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"Tensor on GPU: {x}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Additional system information
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print("\nNVIDIA-SMI output:")
        print(nvidia_smi)
    except Exception as e:
        print("Error running nvidia-smi:", str(e))

    try:
        import subprocess
        nvcc = subprocess.check_output(['nvcc', '--version']).decode()
        print("nvcc --version output:")
        print(nvcc)
    except Exception as e:
        print("Error running nvcc --version:", str(e))
