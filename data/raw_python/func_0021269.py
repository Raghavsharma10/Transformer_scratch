def get_parser():
    """Load parser for command line arguments.

    It parses argv/input into args variable.
    """
    desc = Colors.LIGHTBLUE + textwrap.dedent(
        '''\
        Welcome to
                      _                           _                            _
           __ _ _   _| |_ ___      _ __  _   _   | |_ ___  _ __ _ __ ___ _ __ | |_
          / _` | | | | __/ _ \    | '_ \| | | |  | __/ _ \| '__| '__/ _ \ '_ \| __|
         | (_| | |_| | || (_) |   | |_) | |_| |  | || (_) | |  | | |  __/ | | | |_
          \__,_|\__,_|\__\___/____| .__/ \__, |___\__\___/|_|  |_|  \___|_| |_|\__|
                            |_____|_|    |___/_____|

        ------------------------------------
          auto_py_torrent is an automated tool for download files by obtaining
        torrents or magnets that are in different provided pages that the
        user can choose.

          Its goal is to make it easier for users to find the files they want
        and download them instantly.

          An auto_py_torrent command is provided in which the user can
        currently choose between two modes, best_rated and list mode, then it
        selects one of the torrent tracking pages for multimedia content and
        finally enter the text of what you want to download.
        ------------------------------------
        ''') + Colors.ENDC
    usage_info = Colors.LGREEN + textwrap.dedent(
        '''\

        Use "%(prog)s --help" for more information.
        Examples:
            use "%(prog)s MODE SELECTED_PAGE STRING_TO_SEARCH # generic.
            use "%(prog)s 0 0 "The simpsons" # best rated.
            use "%(prog)s 1 0 "The simpsons" # list rated.

        Mode options:
            0: best_rated. # Download the most rated file.
            1: list. # Get a list, and select one of them.

        Page list options:
            0: torrent project.
            1: the pirate bay.
            2: 1337x.
            3: eztv.
            4: limetorrents.
            5: isohunt.
        ''') + Colors.ENDC
    epi = Colors.LIGHTPURPLE + textwrap.dedent(
        '''\
        -> Thanks for using auto_py_torrent!
        ''') + Colors.ENDC

    # Parent and only parser.
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage_info,
        description=desc,
        epilog=epi)
    parser.add_argument('mode', action='store',
                        choices=range(len(MODES)),
                        type=int,
                        help='Select mode of file download.\n'
                             '    e.g: 0(rated) or 1(list).')
    parser.add_argument('torr_page', action='store',
                        choices=range(len(TORRENTS)),
                        type=int,
                        help='Select tracking page to download from.\n'
                             '    e.g: 0 to .. ' + str(len(TORRENTS)-1) + '.')
    parser.add_argument('str_search', action='store',
                        type=str,
                        help='Input torrent string to search.\n'
                             '    e.g: "String search"')
    return(parser)