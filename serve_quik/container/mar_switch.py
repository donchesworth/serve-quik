import os
from pathlib import Path
import pytorch_quik as pq
from types import SimpleNamespace

BASEPATH = "/workspaces/rdp-vscode-devcontainer"
MODELNAME = "marianmt_es_en"
PROJECTNAME = "text-translate"


def main():
    args = SimpleNamespace(model_type = "marianmt")
    serve_path = Path(f"{BASEPATH}/{PROJECTNAME}/serve/")
    mar_input = serve_path.joinpath("mar", f"{MODELNAME}.mar")
    serve_out = Path(f"{BASEPATH}/torch-serve/{PROJECTNAME}")
    mar_output = serve_out.joinpath("mar")
    mar_input.unlink(missing_ok=True)
    mar_output.joinpath(f"{MODELNAME}.mar").unlink(missing_ok=True)
    pq.serve.create_mar(
        args, serve_path, MODELNAME, handler="text_trans_handler.py")
    os.system(f"cp {mar_input} {mar_output}")


if __name__ == "__main__":
    main()
