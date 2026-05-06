def show(block=False):
    """Show current figures using vispy

    Parameters
    ----------
    block : bool
        If True, blocking mode will be used. If False, then non-blocking
        / interactive mode will be used.

    Returns
    -------
    canvases : list
        List of the vispy canvases that were created.
    """
    if not has_matplotlib():
        raise ImportError('Requires matplotlib version >= 1.2')
    cs = [_mpl_to_vispy(plt.figure(ii)) for ii in plt.get_fignums()]
    if block and len(cs) > 0:
        cs[0].app.run()
    return cs