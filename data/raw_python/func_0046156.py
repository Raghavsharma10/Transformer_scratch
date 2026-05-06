def pull_dependencies(collector, image, **kwargs):
    """Pull an image's dependent images"""
    for dep in image.commands.dependent_images:
        kwargs["image"] = dep
        pull_arbitrary(collector, **kwargs)