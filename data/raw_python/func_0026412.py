def frontend(ctx, dev, rebuild, no_install, build_type):
    """Build and install frontend"""

    install_frontend(instance=ctx.obj['instance'],
                     forcerebuild=rebuild,
                     development=dev,
                     install=not no_install,
                     build_type=build_type)