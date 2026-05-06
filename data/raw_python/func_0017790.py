def data_complete(datadir, sitedir, get_container_name):
    """
    Return True if the directories and containers we're expecting
    are present in datadir, sitedir and containers
    """
    if any(not path.isdir(sitedir + x)
            for x in ('/files', '/run', '/solr')):
        return False

    if docker.is_boot2docker():
        # Inspect returns None if the container doesn't exist.
        return all(docker.inspect_container(get_container_name(x))
                for x in ('pgdata', 'venv'))

    return path.isdir(datadir + '/venv') and path.isdir(sitedir + '/postgres')