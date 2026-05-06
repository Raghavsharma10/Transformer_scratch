def install_all(ctx, clear_all):
    """Default-Install everything installable

    \b
    This includes
    * System user (hfos.hfos)
    * Self signed certificate
    * Variable data locations (/var/lib/hfos and /var/cache/hfos)
    * All the official modules in this repository
    * Default module provisioning data
    * Documentation
    * systemd service descriptor

    It does NOT build and install the HTML5 frontend."""

    _check_root()

    instance = ctx.obj['instance']
    dbhost = ctx.obj['dbhost']
    dbname = ctx.obj['dbname']
    port = ctx.obj['port']

    install_system_user()
    install_cert(selfsigned=True)

    install_var(instance, clear_target=clear_all, clear_all=clear_all)
    install_modules(wip=False)
    install_provisions(provision=None, clear_provisions=clear_all)
    install_docs(instance, clear_target=clear_all)

    install_service(instance, dbhost, dbname, port)
    install_nginx(instance, dbhost, dbname, port)

    log('Done')