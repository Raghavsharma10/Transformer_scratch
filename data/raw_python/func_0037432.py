def configure(config={}, datastore=None, nested=False):
    """
    Useful for when you need to control Switchboard's setup
    """
    if nested:
        config = nested_config(config)
    # Re-read settings to make sure we have everything.
    # XXX It would be really nice if we didn't need to do this.
    Settings.init(**config)

    if datastore:
        Switch.ds = datastore

    # Register the builtins
    __import__('switchboard.builtins')