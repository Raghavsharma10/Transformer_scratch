def run_it():
    """Search and download torrents until the user says it so."""
    initialize()
    parser = get_parser()
    args = None
    first_parse = True
    while(True):
        if first_parse is True:
            first_parse = False
            args = parser.parse_args()
        else:
            print(textwrap.dedent(
                '''\
                Search again like in the beginning.
                  -- You can either choose best rated or list mode.
                  -- This time, you can insert the search string without double quotes.
                  Remember the list mode options!
                    0: torrent project.
                    1: the pirate bay.
                    2: 1337x.
                    3: eztv.
                    4: limetorrents.
                    5: isohunt.
                '''))

            print('Or.. if you want to exit just write "' +
                  Colors.LRED + 'Q' + Colors.ENDC + '" or "' +
                  Colors.LRED + 'q' + Colors.ENDC + '".')
            input_parse = input('>> ').replace("'", "").replace('"', '')
            if input_parse in ['Q', 'q']:
                sys.exit(1)

            args = parser.parse_args(input_parse.split(' ', 2))

        if args.str_search.strip() == "":
            print('Please insert an appropiate non-empty string.')
        else:
            auto = AutoPy(*insert(args))
            auto.get_content()
            auto.select_torrent()
            auto.download_torrent()