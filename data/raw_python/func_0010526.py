def parse_arguments(args=None):
    """
    Parse the program arguments.
    :return: argparse.Namespace object with the parsed arguments
    """
    parser = get_argument_parser()

    # Autocomplete arguments
    autocomplete(parser)

    ns = parser.parse_args(args=args)
    return ArgumentSettings(
        program=ArgumentProgramSettings(
            log=ArgumentLogSettings(
                path=None,
                level=ns.loglevel,
            ),
            settings=ArgumentSettingsSettings(
                path=ns.settings_path,
            ),
            client=ArgumentClientSettings(
                type=ns.client_type,
                cli=ArgumentClientCliSettings(
                    interactive=False,
                ),
                gui=ArgumentClientGuiSettings(
                ),
            ),
        ),
        search=ArgumentSearchSettings(
            recursive=ns.recursive,
            working_directory=ns.video_path,
        ),
        filter=FilterSettings(
            languages=ns.languages,
        ),
        download=DownloadSettings(
            rename_strategy=ns.rename_strategy,
        ),
        providers=ns.providers,
        proxy=ns.proxy,
        test=ns.test,
    )