def rm_raw(ctx, dataset, kwargs):
    "removes the raw unprocessed data"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).rm_raw(**kwargs)