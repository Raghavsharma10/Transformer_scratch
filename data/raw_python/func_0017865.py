def data_only_container(name, volumes):
    """
    create "data-only container" if it doesn't already exist.

    We'd like to avoid these, but postgres + boot2docker make
    it difficult, see issue #5
    """
    info = inspect_container(name)
    if info:
        return
    c = _get_docker().create_container(
        name=name,
        image='datacats/postgres',  # any image will do
        command='true',
        volumes=volumes,
        detach=True)
    return c