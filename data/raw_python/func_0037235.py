def cli(parser):
    '''
    Currently a cop-out -- just calls easy_install
    '''
    parser.add_argument('-n', '--dry-run', action='store_true', help='Print uninstall actions without running')
    parser.add_argument('packages', nargs='+', help='Packages to install')
    opts = parser.parse_args()

    for package in opts.packages:
        install(package, execute=not opts.dry_run)