def reqs(ctx, dataset, kwargs):
    "Get the dataset's pip requirements"

    kwargs = parse_kwargs(kwargs)
    (print)(data(dataset, **ctx.obj).reqs(**kwargs))