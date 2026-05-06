def make_all(collector, **kwargs):
    """Creates all the images in layered order"""
    configuration = collector.configuration
    push = configuration["harpoon"].do_push
    only_pushable = configuration["harpoon"].only_pushable
    if push:
        only_pushable = True

    tag = kwargs.get("artifact", NotSpecified)
    if tag is NotSpecified:
        tag = configuration["harpoon"].tag

    images = configuration["images"]
    for layer in Builder().layered(images, only_pushable=only_pushable):
        for _, image in layer:
            if tag is not NotSpecified:
                image.tag = tag
            Builder().make_image(image, images, ignore_deps=True, ignore_parent=True)
            print("Created image {0}".format(image.image_name))
            if push and image.image_index:
                Syncer().push(image)