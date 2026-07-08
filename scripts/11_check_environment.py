import importlib
import sys


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "cv2",
    "PIL",
    "matplotlib",
    "sklearn",
    "streamlit",
    "yaml",
    "tqdm",
    "torch",
    "torchvision",
]


def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "installed")
        return True, version
    except Exception as exc:
        return False, str(exc)


def main():
    print("FactorySense-R Environment Check")
    print("=" * 40)
    print(f"Python version: {sys.version}")

    all_ok = True

    for package in REQUIRED_PACKAGES:
        ok, result = check_package(package)
        status = "OK" if ok else "MISSING"
        print(f"{package:15} {status:8} {result}")

        if not ok:
            all_ok = False

    try:
        import torch

        print("\nPyTorch device support")
        print("-" * 40)
        print(f"Torch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    except Exception:
        pass

    print("\nResult")
    print("-" * 40)

    if all_ok:
        print("Environment OK")
    else:
        print("Some packages are missing. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()