def show_menu(entries, **kwargs):
    """Shows a menu with the given list of `MenuEntry` items.

    **Params**:
      - header (str) - String to show above menu.
      - note (str) - String to show as a note below menu.
      - msg (str) - String to show below menu.
      - dft (str) - Default value if input is left blank.
      - compact (bool) - If true, the menu items will not be displayed
        [default: False].
      - returns (str) - Controls what part of the menu entry is returned,
        'func' returns function result [default: name].
      - limit (int) - If set, limits the number of menu entries show at a time
        [default: None].
      - fzf (bool) - If true, can enter FCHR at the menu prompt to search menu.
    """
    global _AUTO
    hdr = kwargs.get('hdr', "")
    note = kwargs.get('note', "")
    dft = kwargs.get('dft', "")
    fzf = kwargs.pop('fzf', True)
    compact = kwargs.get('compact', False)
    returns = kwargs.get('returns', "name")
    limit = kwargs.get('limit', None)
    dft = kwargs.get('dft', None)
    msg = []
    if limit:
        return show_limit(entries, **kwargs)
    def show_banner():
        banner = "-- MENU"
        if hdr:
            banner += ": " + hdr
        banner += " --"
        msg.append(banner)
        if _AUTO:
            return
        for i in entries:
            msg.append("  (%s) %s" % (i.name, i.desc))
    valid = [i.name for i in entries]
    if type(dft) == int:
        dft = str(dft)
    if dft not in valid:
        dft = None
    if not compact:
        show_banner()
    if note and not _AUTO:
        msg.append("[!] " + note)
    if fzf:
        valid.append(FCHR)
    msg.append(QSTR + kwargs.get('msg', "Enter menu selection"))
    msg = os.linesep.join(msg)
    entry = None
    while entry not in entries:
        choice = ask(msg, vld=valid, dft=dft, qstr=False)
        if choice == FCHR and fzf:
            try:
                from iterfzf import iterfzf
                choice = iterfzf(reversed(["%s\t%s" % (i.name, i.desc) for i in entries])).strip("\0").split("\t", 1)[0]
            except:
                warn("Issue encountered during fzf search.")
        match = [i for i in entries if i.name == choice]
        if match:
            entry = match[0]
    if entry.func:
        fresult = run_func(entry)
        if "func" == returns:
            return fresult
    try:
        return getattr(entry, returns)
    except:
        return getattr(entry, "name")