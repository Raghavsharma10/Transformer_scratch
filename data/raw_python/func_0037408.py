def load_backend(backend_name):
    """ load pool backend."""
    try:
        if len(backend_name.split(".")) > 1:
            mod = import_module(backend_name)
        else:
            mod = import_module("spamc.backend_%s" % backend_name)
        return mod
    except ImportError:
        error_msg = "%s isn't a spamc backend" % backend_name
        raise ImportError(error_msg)