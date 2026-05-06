def read_login(collector, image, **kwargs):
    """Login to a docker registry with read permissions"""
    docker_api = collector.configuration["harpoon"].docker_api
    collector.configuration["authentication"].login(docker_api, image, is_pushing=False, global_docker=True)