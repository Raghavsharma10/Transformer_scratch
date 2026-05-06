def push_image(registry, image):
    # type: (str, Dict[str, Any]) -> None
    """ Push the given image to selected repository.

    Args:
        registry (str):
            The name of the registry we're pushing to. This is the address of
            the repository without the protocol specification (no http(s)://)
        image (dict[str, Any]):
            The dict containing the information about the image. This is the
            same dictionary as defined in DOCKER_IMAGES variable.
    """
    values = {
        'registry': registry,
        'image': image['name'],
    }

    log.info("Pushing <33>{registry}<35>/{image}".format(**values))
    shell.run('docker push {registry}/{image}'.format(**values))