def stop_supporting_containers(get_container_name, extra_containers):
    """
    Stop postgres and solr containers, along with any specified extra containers
    """
    docker.remove_container(get_container_name('postgres'))
    docker.remove_container(get_container_name('solr'))
    for container in extra_containers:
        docker.remove_container(get_container_name(container))