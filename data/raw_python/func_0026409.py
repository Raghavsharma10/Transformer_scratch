def nginx(ctx, hostname):
    """Install nginx configuration"""

    install_nginx(ctx.obj['dbhost'], ctx.obj['dbname'], ctx.obj['port'], hostname)