def get_args():
    u"""
    ./main --config /etc/blackbird/etc/default.cfg --debug ...
    Return command-line options(arguments).
    """

    description = "The Daemon send various value for zabbix_sender."
    parser = argparse.ArgumentParser(description)

    parser.add_argument('--config', '-c',
                        default='conf/defaults.cfg',
                        help='Specify "defaults.cfg" file'
                        )

    parser.add_argument('--debug-mode', '-d',
                        action='store_true',
                        help='Turn on debug mode',
                        dest='debug_mode'
                        )

    parser.add_argument('--pid-file', '-p',
                        default=os.path.join(
                            os.path.abspath(os.path.curdir),
                            'blackbird.pid'
                        ),
                        help='pid file location',
                        dest='pid_file'
                        )

    parser.add_argument('--foreground', '-f',
                        default=True,
                        action='store_false',
                        help='Turn on foreground mode',
                        dest='detach_process'
                        )

    parser.add_argument('--version', '-V',
                        default=False,
                        action='store_true',
                        help='Show version information',
                        dest='show_version'
                        )

    args = parser.parse_args()
    args.pid_file = is_pid(args.pid_file)

    return parser.parse_args()