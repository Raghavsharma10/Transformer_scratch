def copy_file(
    source_path,
    target_path,
    allow_undo=True,
    no_confirm=False,
    rename_on_collision=True,
    silent=False,
    extra_flags=0,
    hWnd=None
):
    """Perform a shell-based file copy. Copying in
    this way allows the possibility of undo, auto-renaming,
    and showing the "flying file" animation during the copy.

    The default options allow for undo, don't automatically
    clobber on a name clash, automatically rename on collision
    and display the animation.
    """
    return _file_operation(
        shellcon.FO_COPY,
        source_path,
        target_path,
        allow_undo,
        no_confirm,
        rename_on_collision,
        silent,
        extra_flags,
        hWnd
    )