def rm_subsets(ctx, dataset, kwargs):
    "removes the dataset's training-set and test-set folders if they exists"

    kwargs = parse_kwargs(kwargs)
    data(dataset, **ctx.obj).rm_subsets(**kwargs)