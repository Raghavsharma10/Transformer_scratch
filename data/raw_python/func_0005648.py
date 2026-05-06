def build_images():
    # type: () -> None
    """ Build all docker images for the project. """
    registry = conf.get('docker.registry')
    docker_images = conf.get('docker.images', [])

    for image in docker_images:
        build_image(registry, image)