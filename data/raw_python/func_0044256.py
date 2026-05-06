def ls(ctx, available):
    "List installed datasets on path"

    path = ctx.obj['path']
    global_ = ctx.obj['global_']

    _ls(available=available, **ctx.obj)