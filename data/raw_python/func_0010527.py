def get_argument_parser():
    """
    Get a parser that is able to parse program arguments.
    :return: instance of arparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=project.get_description(),
                                     epilog=_('Visit us at {website}.').format(website=project.WEBSITE_MAIN))

    parser.add_argument('--version', action='version',
                        version='{project} {version}'.format(project=project.PROJECT_TITLE,
                                                             version=project.PROJECT_VERSION_STR))
    parser.add_argument('-T', '--test', dest='test',
                        action='store_true', default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument('-V', '--video', dest='video_path', default=None, metavar='PATH',
                        nargs=argparse.ONE_OR_MORE, action=PathsAction,
                        help=_('Full path to your video(s).'))
    parser.add_argument('-s', '--settings', dest='settings_path', type=Path, default=None, metavar='FILE',
                        help=_('Set the settings file.'))
    parser.add_argument('-l', '--lang', dest='languages', metavar='LANGUAGE',
                        default=[UnknownLanguage.create_generic()],
                        nargs=argparse.ONE_OR_MORE, action=LanguagesAction,
                        help=_('Set the preferred subtitle language(s) for download and upload.'))

    # interface options
    interface_group = parser.add_argument_group(_('interface'), _('Change settings of the interface'))
    guicli = interface_group.add_mutually_exclusive_group()
    guicli.add_argument('-g', '--gui', dest='client_type',
                        action='store_const', const=ClientType.GUI,
                        help=_('Run application in GUI mode. This is the default.'))
    guicli.add_argument('-c', '--cli', dest='client_type',
                        action='store_const', const=ClientType.CLI,
                        help=_('Run application in CLI mode.'))
    parser.set_defaults(client_type=ClientType.GUI)

    # logger options
    loggroup = parser.add_argument_group(_('logging'), _('Change the amount of logging done.'))
    loglvlex = loggroup.add_mutually_exclusive_group()
    loglvlex.add_argument('-d', '--debug', dest='loglevel',
                          action='store_const', const=logging.DEBUG,
                          help=_('Print log messages of debug severity and higher to stderr.'))
    loglvlex.add_argument('-w', '--warning', dest='loglevel',
                          action='store_const', const=logging.WARNING,
                          help=_('Print log messages of warning severity and higher to stderr. This is the default.'))
    loglvlex.add_argument('-e', '--error', dest='loglevel',
                          action='store_const', const=logging.ERROR,
                          help=_('Print log messages of error severity and higher to stderr.'))
    loglvlex.add_argument('-q', '--quiet', dest='loglevel',
                          action='store_const', const=LOGGING_LOGNOTHING,
                          help=_('Don\'t log anything to stderr.'))
    loggroup.set_defaults(loglevel=logging.WARNING)

    loggroup.add_argument('--log', dest='logfile', metavar='FILE', type=Path,
                          default=None, help=_('Path name of the log file.'))

    # cli options
    cli_group = parser.add_argument_group(_('cli'), _('Change the behavior of the command line interface.'))
    cli_group.add_argument('-i', '--interactive', dest='interactive',
                           action='store_true', default=False,
                           help=_('Prompt user when decisions need to be done.'))
    cli_group.add_argument('-r', '--recursive', dest='recursive',
                           action='store_true', default=False,
                           help=_('Search for subtitles recursively.'))

    operation_group = cli_group.add_mutually_exclusive_group()
    operation_group.add_argument('-D', '--download', dest='operation', action='store_const', const=CliAction.DOWNLOAD,
                                 help=_('Download subtitle(s). This is the default.'))
    operation_group.add_argument('-U', '--upload', dest='operation', action='store_const', const=CliAction.UPLOAD,
                                 help=_('Upload subtitle(s).'))
    # operation_group.add_argument('-L', '--list', dest='operation', action='store_const', const=CliAction.LIST,
    #                              help=_('List available subtitle(s) without downloading.'))
    parser.set_defaults(operation=CliAction.DOWNLOAD)

    rename_group = cli_group.add_mutually_exclusive_group()
    rename_group.add_argument('--rename-online', dest='rename_strategy', action='store_const',
                              const=SubtitleRenameStrategy.ONLINE,
                              help=_('Use the on-line subtitle filename as name for the downloaded subtitles. '
                                     'This is the default.'))
    rename_group.add_argument('--rename-video', dest='rename_strategy', action='store_const',
                              const=SubtitleRenameStrategy.VIDEO,
                              help=_('Use the local video filename as name for the downloaded subtitle.'))
    rename_group.add_argument('--rename-lang', dest='rename_strategy', action='store_const',
                              const=SubtitleRenameStrategy.VIDEO_LANG,
                              help=_('Use the local video filename + language as name for the downloaded subtitle.'))
    rename_group.add_argument('--rename-uploader', dest='rename_strategy', action='store_const',
                              const=SubtitleRenameStrategy.VIDEO_LANG_UPLOADER,
                              help=_('Use the local video filename + uploader + language '
                                     'as name for the downloaded subtitle.'))
    parser.set_defaults(rename_strategy=SubtitleRenameStrategy.ONLINE)

    # online options
    online_group = parser.add_argument_group('online', 'Change parameters related to the online provider.')
    online_group.add_argument('-P', '--proxy', dest='proxy', default=None, action=ProxyAction,
                              help=_('Proxy to use on internet connections.'))
    online_group.add_argument('--provider', dest='providers', metavar='NAME [KEY1=VALUE1 [KEY2=VALUE2 [...]]]',
                              nargs=argparse.ONE_OR_MORE, default=None, action=ProviderAction,
                              help=_('Enable and configure a provider.'))

    return parser