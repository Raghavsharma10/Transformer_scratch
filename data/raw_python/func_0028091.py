def main():
    """
    Entry point for choosing what subcommand to run. Really should be using asciidocapi
    """
    # Try parsing command line args and flags with docopt
    args = docopt(__doc__, version="cdk")
    # Am I going to need validation? No Schema for the moment...
    if args['FILE']:
        out = output_file(args['FILE'])
        # Great! Run asciidoc with appropriate flags
        theme = pick_theme(args['--theme'])
        if theme not in listdir(THEMES_DIR):
            exit('Selected theme "%s" not found. Check ~/.cdk/prefs' % theme)
        cmd = create_command(theme, args['--bare'], args['--toc'], args['--notransition'],
                             args['--logo'])
        run_command(cmd, args)
        if args['--toc']:
            add_css(out, '.deck-container .deck-toc li a span{color: #888;display:inline;}')
        if args['--custom-css']:
            add_css_file(out, args['--custom-css'])
        if args['--open']:
            webbrowser.open("file://" + abspath(out))

    # other commands
    elif args['--generate']:
        if isfile(args['--generate']):
            exit("%s already exists!" % args['--generate'])
        with open(args['--generate'], "w") as fp:
            sample = join(LOCATION,  "custom", "sample.asc")
            fp.write(open(sample).read())
            print("Created sample slide deck in %s..." % args['--generate'])
        exit()

    elif args['--install-theme']:
        path = args['--install-theme']
        if not isfile(path):
            exit("Theme file not found.")
        if not path.endswith(".zip"):
            exit("Theme installation currently only supports theme install from "
                 ".zip files.")
        install_theme(path)
    elif args['--default-theme']:
        set_default_theme(args['--default-theme'])