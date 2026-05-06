def prepare_input(argv=None):
    """
    Get, parse and prepare input file.
    """
    p = ArgumentParser(description='InsiliChem Ommprotocol: '
                       'easy to deploy MD protocols for OpenMM')
    p.add_argument('input', metavar='INPUT FILE', type=extant_file,
                   help='YAML input file')
    p.add_argument('--version', action='version', version='%(prog)s v{}'.format(__version__))
    p.add_argument('-c', '--check', action='store_true',
                   help='Validate input file only')
    args = p.parse_args(argv if argv else sys.argv[1:])

    jinja_env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    # Load config file
    with open(args.input) as f:
        rendered = jinja_env.from_string(f.read()).render()
        cfg = yaml.load(rendered, Loader=YamlLoader)
    # Paths and dirs
    from .md import SYSTEM_OPTIONS
    cfg['_path'] = os.path.abspath(args.input)
    cfg['system_options'] = prepare_system_options(cfg, defaults=SYSTEM_OPTIONS)
    cfg['outputpath'] = sanitize_path_for_file(cfg.get('outputpath', '.'), args.input)

    if not args.check:
        with ignored_exceptions(OSError):
            os.makedirs(cfg['outputpath'])

    handler = prepare_handler(cfg)

    return handler, cfg, args