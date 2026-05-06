def rm(ctx, dataset, kwargs):
    "removes the dataset's folder if it exists"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).rm(**kwargs)