def delete_file(
    source_path,
    allow_undo=True,
    no_confirm=False,
    silent=False,
    extra_flags=0,
    hWnd=None
):
    """Perform a shell-based file delete. Deleting in
    this way uses the system recycle bin, allows the
    possibility of undo, and showing the "flying file"
    animation during the delete.

    The default options allow for undo, don't automatically
    clobber on a name clash and display the animation.
    """
    return _file_operation(
        shellcon.FO_DELETE,
        source_path,
        None,
        allow_undo,
        no_confirm,
        False,
        silent,
        extra_flags,
        hWnd
    )