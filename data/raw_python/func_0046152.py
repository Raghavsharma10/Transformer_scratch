def push(collector, image, **kwargs):
    """Push an image"""
    if not image.image_index:
        raise BadOption("The chosen image does not have a image_index configuration", wanted=image.name)
    tag = kwargs["artifact"]
    if tag is NotSpecified:
        tag = collector.configuration["harpoon"].tag
    if tag is not NotSpecified:
        image.tag = tag
    Builder().make_image(image, collector.configuration["images"], pushing=True)
    Syncer().push(image)