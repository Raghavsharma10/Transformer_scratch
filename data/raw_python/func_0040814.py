def ErrorMessage(text, **kwargs):
    """Show an error message dialog to the user.
    
    This will raise a Zenity Error Dialog with a description of the error.
    
    text - A description of the error.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = ['--text=%s' % text]
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    run_zenity('--error', *args).wait()