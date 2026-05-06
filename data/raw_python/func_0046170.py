def untag(collector, image, artifact, **kwargs):
    """Tag an image!"""
    if artifact in (None, "", NotSpecified):
        artifact = collector.configuration["harpoon"].tag

    if artifact is NotSpecified:
        raise BadOption("Please specify a tag using the artifact or tag options")

    image.tag = artifact
    image_name = image.image_name_with_tag

    log.info("Removing image\timage={0}".format(image_name))
    try:
        image.harpoon.docker_api.remove_image(image_name)
    except docker.errors.ImageNotFound:
        log.warning("No image was found to remove")