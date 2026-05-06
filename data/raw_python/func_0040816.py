def GetText(text='', entry_text='', password=False, **kwargs):
    """Get some text from the user.

    This will raise a Zenity Text Entry Dialog.  It returns the text the user 
    entered or None if the user hit cancel.

    text - A description of the text to enter.
    entry_text - The initial value of the text entry box.
    password - True if text entered should be hidden by stars.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    if text:
        args.append('--text=%s' % text)
    if entry_text:
        args.append('--entry-text=%s' % entry_text)
    if password:
        args.append('--hide-text')

    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--entry', *args)

    if p.wait() == 0:
        return p.stdout.read()[:-1]