def make(collector, image, **kwargs):
    """Just create an image"""
    tag = kwargs.get("artifact", NotSpecified)
    if tag is NotSpecified:
        tag = collector.configuration["harpoon"].tag

    if tag is not NotSpecified:
        image.tag = tag

    Builder().make_image(image, collector.configuration["images"])
    print("Created image {0}".format(image.image_name))