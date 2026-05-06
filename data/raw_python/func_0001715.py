def main():
    """
    Entry point
    """
    rc_settings = read_rcfile()
    parser = ArgumentParser(description='Millipede generator')
    parser.add_argument('-s', '--size',
                        type=int,
                        nargs="?",
                        help='the size of the millipede')
    parser.add_argument('-c', '--comment',
                        type=str,
                        help='the comment')
    parser.add_argument('-v', '--version',
                        action='version',
                        version=__version__)
    parser.add_argument('-r', '--reverse',
                        action='store_true',
                        help='reverse the millipede')
    parser.add_argument('-t', '--template',
                        help='customize your millipede')
    parser.add_argument('-p', '--position',
                        type=int,
                        help='move your millipede')
    parser.add_argument('-o', '--opposite',
                        action='store_true',
                        help='go the opposite direction')
    parser.add_argument(
        '--http-host',
        metavar="The http server to send the data",
        help='Send the millipede via an http post request'
    )
    parser.add_argument(
        '--http-auth',
        metavar='user:pass',
        help='Used to authenticate to the API ',
        default=os.environ.get('HTTP_AUTH')
    )
    parser.add_argument(
        '--http-data',
        metavar='key=value',
        nargs='*',
        help='Add additional HTTP POST data'
    )
    parser.add_argument(
        '--http-name',
        metavar='name',
        help='The json variable name that will contain the millipede'
    )

    args = parser.parse_args()

    settings = compute_settings(vars(args), rc_settings)

    out = millipede(
        settings['size'],
        comment=settings['comment'],
        reverse=settings['reverse'],
        template=settings['template'],
        position=settings['position'],
        opposite=settings['opposite']
    )

    if args.http_host:
        if args.http_auth:
            try:
                login, passwd = args.http_auth.split(':')
            except ValueError:
                parser.error(
                    "Credentials should be a string like "
                    "`user:pass'"
                )
        else:
            login = None
            passwd = None

        api_post(
            out,
            args.http_host,
            args.http_name,
            http_data=args.http_data,
            auth=(login, passwd)
        )

    print(out, end='')