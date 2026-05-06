def push_images():
    # type: () -> None
    """ Push all project docker images to a remote registry. """
    registry = conf.get('docker.registry')
    docker_images = conf.get('docker.images', [])

    if registry is None:
        log.err("You must define docker.registry conf variable to push images")
        sys.exit(-1)

    for image in docker_images:
        push_image(registry, image)