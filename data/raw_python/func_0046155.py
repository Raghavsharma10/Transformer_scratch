def pull(collector, image, **kwargs):
    """Pull an image"""
    if not image.image_index:
        raise BadOption("The chosen image does not have a image_index configuration", wanted=image.name)
    tag = kwargs["artifact"]
    if tag is NotSpecified:
        collector.configuration["harpoon"].tag
    if tag is not NotSpecified:
        image.tag = tag
        log.info("Pulling tag: %s", tag)
    Syncer().pull(image, ignore_missing=image.harpoon.ignore_missing)