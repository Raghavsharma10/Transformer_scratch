def saveJSON(g, data, backup=False):
    """
    Saves the current setup to disk.

    g : hcam_drivers.globals.Container
    Container with globals

    data : dict
    The current setup in JSON compatible dictionary format.

    backup : bool
    If we are saving a backup on close, don't prompt for filename
    """
    if not backup:
        fname = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('json files', '.json'), ],
            initialdir=g.cpars['app_directory']
            )
    else:
        fname = os.path.join(os.path.expanduser('~/.hdriver'), 'app.json')

    if not fname:
        g.clog.warn('Aborted save to disk')
        return False

    with open(fname, 'w') as of:
        of.write(
            json.dumps(data, sort_keys=True, indent=4,
                       separators=(',', ': '))
        )
    g.clog.info('Saved setup to' + fname)
    return True