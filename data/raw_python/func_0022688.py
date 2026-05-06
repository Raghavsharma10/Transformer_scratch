def load_ipython_extension(ipython):
    """ Entry point of the IPython extension

    Parameters
    ----------

    IPython : IPython interpreter
        An instance of the IPython interpreter that is handed
        over to the extension
    """
    import IPython

    # don't continue if IPython version is < 3.0
    ipy_version = LooseVersion(IPython.__version__)
    if ipy_version < LooseVersion("3.0.0"):
        ipython.write_err("Your IPython version is older than "
                          "version 3.0.0, the minimum for Vispy's"
                          "IPython backend. Please upgrade your IPython"
                          "version.")
        return

    _load_webgl_backend(ipython)