def main(args=None):
    """The main."""
    parser = create_parser()
    args = vars(parser.parse_args(args))

    log_mapping = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    log_level = log_mapping[args['logging']]
    del(args['logging'])

    global logger
    logger = configure_logging(log_level)

    args_mapping = {
        "path": "local_path",
        "remote": "remote_url",
        "ssh_config": "ssh_config_path",
        "exclude_from": "exclude_file",
        "known_hosts": "known_hosts_path",
        "do_not_delete": "delete",
        "key": "identity_files",
    }

    kwargs = {  # convert the argument names to class constructor parameters
        args_mapping[k]: v
        for k, v in args.items()
        if v and k in args_mapping
    }

    kwargs.update({
        k: v
        for k, v in args.items()
        if v and k not in args_mapping
    })

    # Special case: disable known_hosts check
    if args['disable_known_hosts']:
        kwargs['known_hosts_path'] = None
        del(kwargs['disable_known_hosts'])

    # Toggle `do_not_delete` flag
    if "delete" in kwargs:
        kwargs["delete"] = not kwargs["delete"]

    # Manually set the default identity file.
    kwargs["identity_files"] = kwargs.get("identity_files", None) or ["~/.ssh/id_rsa"]

    sync = SFTPClone(
        **kwargs
    )
    sync.run()