def service(ctx):
    """Install systemd service configuration"""

    install_service(ctx.obj['instance'], ctx.obj['dbhost'], ctx.obj['dbname'], ctx.obj['port'])