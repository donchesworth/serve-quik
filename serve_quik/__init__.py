from importlib import import_module

__all__ = [
    "api",
    "arg",
    "container",
    "mar",
    "utils",
]
__version__ = "0.0.2"

for submodule in __all__:
    import_module(f"serve_quik.{submodule}")
