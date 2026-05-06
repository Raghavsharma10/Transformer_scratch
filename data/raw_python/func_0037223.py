def cli(parser):
    '''
    Simply calls out to easy_install --upgrade
    '''
    parser.add_argument('packages', nargs='*', default=local_packages(),
        help='Packages to upgrade (defaults to all installed packages)')
    parser.add_argument('-n', '--dry-run', action='store_true',
        help='Print upgrade actions without running')
    opts = parser.parse_args()

    for package in opts.packages:
        upgrade(package, execute=not opts.dry_run)