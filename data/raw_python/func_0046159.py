def pull_all_external(collector, **kwargs):
    """Pull all the external dependencies of all the images"""
    deps = set()

    images = collector.configuration["images"]
    for layer in Builder().layered(images):
        for image_name, image in layer:
            for dep in image.commands.external_dependencies:
                deps.add(dep)

    for dep in sorted(deps):
        kwargs["image"] = dep
        pull_arbitrary(collector, **kwargs)