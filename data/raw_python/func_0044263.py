def rm_compressed(ctx, dataset, kwargs):
    "removes the compressed files"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).rm_compressed(**kwargs)