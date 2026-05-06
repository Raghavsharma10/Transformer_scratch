def load(theTask, canExecute=True, strict=True, defaults=False):
    """ Shortcut to load TEAL .cfg files for non-GUI access where
    loadOnly=True. """
    return teal(theTask, parent=None, loadOnly=True, returnAs="dict",
                canExecute=canExecute, strict=strict, errorsToTerm=True,
                defaults=defaults)