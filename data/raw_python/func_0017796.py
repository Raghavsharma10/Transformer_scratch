def containers_running(get_container_name):
    """
    Return a list of containers tracked by this environment that are running
    """
    running = []
    for n in ['web', 'postgres', 'solr', 'datapusher', 'redis']:
        info = docker.inspect_container(get_container_name(n))
        if info and not info['State']['Running']:
            running.append(n + '(halted)')
        elif info:
            running.append(n)
    return running