def cli(parser):
    '''
    Uninstall inactive Python packages from all accessible site-packages directories.

    Inactive Python packages
    when multiple packages with the same name are installed
    '''
    parser.add_argument('-n', '--dry-run', action='store_true', help='Print cleanup actions without running')
    opts = parser.parse_args()

    for sitedir in site.getsitepackages():
        cleanup(sitedir, execute=not opts.dry_run, verbose=opts.verbose or opts.dry_run)