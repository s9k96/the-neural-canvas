import torch

print("=" * 50)
print("PyTorch Info")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"Interop threads: {torch.get_num_interop_threads()}")

if torch.cuda.is_available():
    device = torch.cuda.current_device()

    print("\nGPU Info")
    print("-" * 50)
    print(f"Device ID: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")

    props = torch.cuda.get_device_properties(device)

    print(f"Total GPU Memory: {props.total_memory / 1024**3:.2f} GB")

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    print(f"Allocated Memory: {allocated / 1024**2:.2f} MB")
    print(f"Reserved Memory: {reserved / 1024**2:.2f} MB")
    print(f"Free (approx): {(props.total_memory - reserved) / 1024**3:.2f} GB")

else:
    print("\nNo CUDA GPU detected. Running on CPU.")
