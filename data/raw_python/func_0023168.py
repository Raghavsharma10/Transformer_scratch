def set_interactive(enabled=True, app=None):
    """Activate the IPython hook for VisPy.  If the app is not specified, the
    default is used.
    """
    if enabled:
        inputhook_manager.enable_gui('vispy', app)
    else:
        inputhook_manager.disable_gui()