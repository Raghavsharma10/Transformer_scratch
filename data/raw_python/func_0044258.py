def size(ctx, dataset, kwargs):
    "Show dataset size"

    kwargs = parse_kwargs(kwargs)
    (print)(data(dataset, **ctx.obj).get(**kwargs).complete_set.size)