def process(ctx, dataset, kwargs):
    "processes the data to a friendly format"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).process(**kwargs)