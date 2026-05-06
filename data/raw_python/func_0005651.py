def build_image(registry, image):
    # type: (str, Dict[str, Any]) -> None
    """ Build docker image.

    Args:
        registry (str):
            The name of the registry this image belongs to. If not given, the
            resulting image will have a name without the registry.
        image (dict[str, Any]):
            The dict containing the information about the built image. This is
            the same dictionary as defined in DOCKER_IMAGES variable.
    """
    if ':' in image['name']:
        _, tag = image['name'].split(':', 1)
    else:
        _, tag = image['name'], None

    values = {
        'registry': '' if registry is None else registry + '/',
        'image': image['name'],
        'tag': tag,
    }

    if tag is None:
        args = [
            '-t {registry}{image}'.format(**values),
            '-t {registry}{image}:{version}'.format(
                version=versioning.current(),
                **values
            ),
        ]
    else:
        args = ['-t {registry}{image}'.format(**values)]

    if 'file' in image:
        args.append('-f {}'.format(conf.proj_path(image['file'])))

    with conf.within_proj_dir(image.get('path', '.')):
        log.info("Building <33>{registry}<35>/{image}", **values)
        shell.run('docker build {args} .'.format(args=' '.join(args)))