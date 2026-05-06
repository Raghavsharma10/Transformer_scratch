def write_login(collector, image, **kwargs):
    """Login to a docker registry with write permissions"""
    docker_api = collector.configuration["harpoon"].docker_api
    collector.configuration["authentication"].login(docker_api, image, is_pushing=True, global_docker=True)