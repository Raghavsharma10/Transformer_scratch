def extract(ctx, dataset, kwargs):
    "extracts the files from the compressed archives"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).extract(**kwargs)