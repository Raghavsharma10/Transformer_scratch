def _load_webgl_backend(ipython):
    """ Load the webgl backend for the IPython notebook"""

    from .. import app
    app_instance = app.use_app("ipynb_webgl")

    if app_instance.backend_name == "ipynb_webgl":
        ipython.write("Vispy IPython module has loaded successfully")
    else:
        # TODO: Improve this error message
        ipython.write_err("Unable to load webgl backend of Vispy")