import pytorch_quik as pq
from serve_quik import arg, container, mar, utils

# SOURCE_LANGS = ['ja', 'de', 'es', 'fr', 'bzs', 'zh', 'ko']
# python main.py -p "text-translate" -mt marianmt -src ja de es -tgt en


def main():
    # args = arg.parse_args([
    #     "-p", "test-project", "-mt", "marianmt", "-src", "es",
    #     "-tgt", "en"
    #     ])

    args = arg.parse_args()

    serve_dir = utils.set_serve_dir(args.project_name)
    container.build_dot_env(serve_dir)

    for src in args.source:
        args.kwargs['source'] = src
        model_dir = utils.set_model_dir(
            serve_dir, args.model_type, args.kwargs
            )
        mar.save_setup_config(model_dir, args)
        tknzr = pq.hugging.get_tokenizer(
            args.model_type, serve_dir.joinpath("tmp"), args.kwargs
            )
        pq.hugging.save_tokenizer(tknzr, model_dir)
        model = pq.hugging.get_pretrained_model(
            args.model_type, kwargs=args.kwargs
            )
        pq.hugging.save_pretrained_model(
            model, serve_path=model_dir, state_dict=model.state_dict()
            )
        mar.create_mar(args.model_type, model_dir)

    container.start_container(serve_dir)


if __name__ == "__main__":
    main()
