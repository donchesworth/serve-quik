from importlib import import_module

__all__ = [
    "mar",
    "utils",
]
__version__ = "0.0.0"

for submodule in __all__:
    import_module(f"serve_quik.{submodule}")
