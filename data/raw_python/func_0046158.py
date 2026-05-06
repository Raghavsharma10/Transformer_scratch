def pull_all(collector, image, **kwargs):
    """Pull all the images"""
    images = collector.configuration["images"]

    for layer in Builder().layered(images, only_pushable=True):
        for image_name, image in layer:
            log.info("Pulling %s", image_name)
            pull(collector, image, **kwargs)