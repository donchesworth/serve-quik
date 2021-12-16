import pytorch_quik as pq
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Optional

KWARGS = ["source", "target"]


def parse_args(arglist: Optional[list] = None) -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-mt", "--model_type")
    parser.add_argument("-src", "--source")
    parser.add_argument("-tgt", "--target")
    parser.add_argument("-hdlr", "--handler")
    args = parser.parse_args(arglist)
    args.kwargs = {k: vars(args)[k] for k in KWARGS if vars(args)[k]}
    return args


def main():
    # sample_args = [
    #     "-m", "text-translate", "-mt", "marianmt", "-src", "es", "-tgt", "en",
    #     "-hdlr", "text_handler.py"
    # ]
    args = parse_args()
    serve_path, tmp_path = pq.io.serve_str(model_name=args.model_name)
    pq.serve.save_setup_config(serve_path, args)
    tknzr = pq.hugging.get_tokenizer(
        args.model_type, tmp_path, args.kwargs
        )
    pq.hugging.save_tokenizer(tknzr, serve_path.parent)
    model = pq.hugging.get_pretrained_model(
        args.model_type, kwargs=args.kwargs
        )
    pq.hugging.save_pretrained_model(
        model, serve_path=serve_path, state_dict=model.state_dict()
        )
    pq.serve.create_mar(
        args, serve_path, handler=args.handler
        )


if __name__ == "__main__":
    main()
