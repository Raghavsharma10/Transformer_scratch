def insert(args):
    """Insert args values into instance variables."""
    string_search = args.str_search
    mode_search = MODES[args.mode]
    page = list(TORRENTS[args.torr_page].keys())[0]
    key_search = TORRENTS[args.torr_page][page]['key_search']
    torrent_page = TORRENTS[args.torr_page][page]['page']
    domain = TORRENTS[args.torr_page][page]['domain']
    return([args, string_search, mode_search, page,
            key_search, torrent_page, domain])