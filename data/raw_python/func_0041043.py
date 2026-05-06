def process_parsed_args(opts: Namespace, error_fun: Optional[Callable], connect: bool=True) -> Namespace:
    """
    Set the defaults for the crc and ontology schemas
    :param opts: parsed arguments
    :param error_fun: Function to call if error
    :param connect: actually connect. (For debugging)
    :return: namespace with additional elements added
    """
    def setdefault(vn: str, default: object) -> None:
        assert vn in opts, "Unknown option"
        if not getattr(opts, vn):
            setattr(opts, vn, default)

    if error_fun and \
            (getattr(opts, 'dburl') is None or getattr(opts, 'user') is None or getattr(opts, 'password') is None):
        error_fun("db url, user id and password must be supplied")
    setdefault('crcdb', opts.dburl)
    setdefault('crcuser', opts.user)
    setdefault('crcpassword', opts.password)
    setdefault('ontodb', opts.dburl)
    setdefault('ontouser', opts.user)
    setdefault('ontopassword', opts.password)
    if connect:
        opts.tables = I2B2Tables(opts)

    # TODO: This approach needs to be re-thought.  As i2b2tablenames is a singleton, any changes here
    # impact the entire testing harness
    if opts.onttable:
        i2b2tablenames.ontology_table = opts.onttable
    return opts