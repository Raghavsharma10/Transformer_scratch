def Routers(typ, share, handler=RoutersHandler):
    """
    Pass the result of this function to the handler argument
    in your attribute declaration
    """
    _sharing_id, _mode = tuple(share.split(":"))
    _router_cls = ROUTERS.get(typ)
    class _Handler(handler):
        mode=_mode
        sharing_id=_sharing_id
        router_cls=_router_cls
    return _Handler