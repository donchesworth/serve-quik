from pathlib import Path
from types import SimpleNamespace
from serve_quik import mar

BASEPATH = "/workspaces/rdp-vscode-devcontainer/serve-quik/deployments"
MODELNAME = "opus-mt-es-en"
PROJECTNAME = "test-project"


def main():
    args = SimpleNamespace(model_type="marianmt")
    model_dir = Path(BASEPATH, PROJECTNAME, MODELNAME)
    # mar_input = model_dir.joinpath("mar", f"{MODELNAME}.mar")
    # serve_out = Path(f"{BASEPATH}/torch-serve/{PROJECTNAME}")
    # mar_output = serve_out.joinpath("mar")
    # mar_input.unlink(missing_ok=True)
    # mar_output.joinpath(f"{MODELNAME}.mar").unlink(missing_ok=True)
    mar.create_mar(args, model_dir)
    # os.system(f"cp {mar_input} {mar_output}")


if __name__ == "__main__":
    main()
