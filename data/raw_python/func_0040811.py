def GetSavename(default=None, **kwargs):
    """Prompt the user for a filename to save as.
    
    This will raise a Zenity Save As Dialog.  It will return the name to save 
    a file as or None if the user hit cancel.
    
    default - The default name that should appear in the save as dialog.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = ['--save']
    if default:
        args.append('--filename=%s' % default)
    
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--file-selection', *args)

    if p.wait() == 0:
        return p.stdout.read().strip().split('|')