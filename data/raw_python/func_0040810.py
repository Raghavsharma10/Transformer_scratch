def GetDirectory(multiple=False, selected=None, sep=None, **kwargs):
    """Prompt the user for a directory.
    
    This will raise a Zenity Directory Selection Dialog.  It will return a 
    list with the selected directories or None if the user hit cancel.
    
    multiple - True to allow the user to select multiple directories.
    selected - Path to the directory to be selected on startup.
    sep - Token to use as the path separator when parsing Zenity's return 
          string.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = ['--directory']
    if multiple:
        args.append('--multiple')
    if selected:
        if not path.lexists(selected):
            raise ValueError("File %s does not exist!" % selected)
        args.append('--filename=%s' % selected)
    if sep:
        args.append('--separator=%s' % sep)
    
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--file-selection', *args)

    if p.wait() == 0:
        return p.stdout.read().strip().split('|')