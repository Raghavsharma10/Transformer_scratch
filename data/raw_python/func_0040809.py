def GetFilename(multiple=False, sep='|', **kwargs):
    """Prompt the user for a filename.
    
    This will raise a Zenity File Selection Dialog. It will return a list with 
    the selected files or None if the user hit cancel.
    
    multiple - True to allow the user to select multiple files.
    sep - Token to use as the path separator when parsing Zenity's return 
          string.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    if multiple:
        args.append('--multiple')
    if sep != '|':
        args.append('--separator=%s' % sep)
    
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--file-selection', *args)

    if p.wait() == 0:
        return p.stdout.read()[:-1].split('|')