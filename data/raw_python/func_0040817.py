def TextInfo(filename=None, editable=False, **kwargs):
    """Show the text of a file to the user.

    This will raise a Zenity Text Information Dialog presenting the user with 
    the contents of a file.  It returns the contents of the text box.

    filename - The path to the file to show.
    editable - True if the text should be editable.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    if filename:
        args.append('--filename=%s' % filename)
    if editable:
        args.append('--editable')

    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--text-info', *args)

    if p.wait() == 0:
        return p.stdout.read()