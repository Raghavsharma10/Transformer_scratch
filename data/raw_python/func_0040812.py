def Notification(text=None, window_icon=None, **kwargs):
    """Put an icon in the notification area.
    
    This will put an icon in the notification area and return when the user
    clicks on it.
    
    text - The tooltip that will show when the user hovers over it.
    window_icon - The stock icon ("question", "info", "warning", "error") or 
                  path to the icon to show.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = []
    if text:
        args.append('--text=%s' % text)
    if window_icon:
        args.append('--window-icon=%s' % window_icon)
    
    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--notification', *args)
    p.wait()