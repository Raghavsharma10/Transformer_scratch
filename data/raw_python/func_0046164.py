def delete_untagged(collector, **kwargs):
    """Find the untagged images and remove them"""
    configuration = collector.configuration
    docker_api = configuration["harpoon"].docker_api
    images = docker_api.images()
    found = False
    for image in images:
        if image["RepoTags"] == ["<none>:<none>"]:
            found = True
            image_id = image["Id"]
            log.info("Deleting untagged image\thash=%s", image_id)
            try:
                docker_api.remove_image(image["Id"])
            except DockerAPIError as error:
                log.error("Failed to delete image\thash=%s\terror=%s", image_id, error)

    if not found:
        log.info("Didn't find any untagged images to delete!")