def freeze(ctx, version: str, clean: bool):
    """
    Freeze current package into a single file
    """
    if clean:
        _clean_spec()
    ctx.invoke(epab.cmd.compile_qt_resources)
    _freeze(version)