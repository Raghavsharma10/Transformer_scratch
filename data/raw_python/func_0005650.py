def docker_list(registry_pass):
    # type: (str) -> None
    """ List docker images stored in the remote registry.

    Args:
        registry_pass (str):
            Remote docker registry password.
    """
    registry = conf.get('docker.registry', None)

    if registry is None:
        log.err("You must define docker.registry conf variable to list images")
        sys.exit(-1)

    registry_user = conf.get('docker.registry_user', None)

    if registry_user is None:
        registry_user = click.prompt("Username")

    rc = client.RegistryClient(registry, registry_user, registry_pass)
    images = {x: rc.list_tags(x) for x in rc.list_images()}

    shell.cprint("<32>Images in <34>{} <32>registry:", registry)
    for image, tags in images.items():
        shell.cprint('  <92>{}', image)
        for tag in tags:
            shell.cprint('      <90>{}:<35>{}', image, tag)