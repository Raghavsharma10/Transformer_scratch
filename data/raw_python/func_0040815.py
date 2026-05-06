def Progress(text='', percentage=0, auto_close=False, pulsate=False, **kwargs):
    """Show a progress dialog to the user.
    
    This will raise a Zenity Progress Dialog.  It returns a callback that 
    accepts two arguments.  The first is a numeric value of the percent 
    complete.  The second is a message about the progress.

    NOTE: This function sends the SIGHUP signal if the user hits the cancel 
          button.  You must connect to this signal if you do not want your 
          application to exit.
    
    text - The initial message about the progress.
    percentage - The initial percentage to set the progress bar to.
    auto_close - True if the dialog should close automatically if it reaches 
                 100%.
    pulsate - True is the status should pulsate instead of progress.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    if text:
        args.append('--text=%s' % text)
    if percentage:
        args.append('--percentage=%s' % percentage)
    if auto_close:
        args.append('--auto-close=%s' % auto_close)
    if pulsate:
        args.append('--pulsate=%s' % pulsate)

    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = Popen([zen_exec, '--progress'] + args, stdin=PIPE, stdout=PIPE)

    def update(percent, message=''):
        if type(percent) == float:
            percent = int(percent * 100)
        p.stdin.write(str(percent) + '\n')
        if message:
            p.stdin.write('# %s\n' % message)
        return p.returncode

    return update