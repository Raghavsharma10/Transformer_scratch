def tag(collector, image, artifact, **kwargs):
    """Tag an image!"""
    if artifact in (None, "", NotSpecified):
        raise BadOption("Please specify a tag using the artifact option")

    if image.image_index in (None, "", NotSpecified):
        raise BadOption("Please specify an image with an image_index option")

    tag = image.image_name
    if collector.configuration["harpoon"].tag is not NotSpecified:
        tag = "{0}:{1}".format(tag, collector.configuration["harpoon"].tag)
    else:
        tag = "{0}:latest".format(tag)

    images = image.harpoon.docker_api.images()
    current_tags = chain.from_iterable(image_conf["RepoTags"] for image_conf in images if image_conf["RepoTags"] is not None)
    if tag not in current_tags:
        raise BadOption("Please build or pull the image down to your local cache before tagging it")

    for image_conf in images:
        if image_conf["RepoTags"] is not None:
            if tag in image_conf["RepoTags"]:
                image_id = image_conf["Id"]
                break

    log.info("Tagging {0} ({1}) as {2}".format(image_id, image.image_name, artifact))
    image.harpoon.docker_api.tag(image_id, repository=image.image_name, tag=artifact, force=True)

    image.tag = artifact
    Syncer().push(image)