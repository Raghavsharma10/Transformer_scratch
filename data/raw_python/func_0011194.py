def pass_obj(f):
    """Similar to :func:`pass_context`, but only pass the object on the
    context onwards (:attr:`Context.obj`).  This is useful if that object
    represents the state of a nested system.
    """
    @pass_context
    def new_func(*args, **kwargs):
        ctx = args[0]
        return ctx.invoke(f, ctx.obj, *args[1:], **kwargs)
    return update_wrapper(new_func, f)