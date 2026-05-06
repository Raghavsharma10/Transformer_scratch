def create_virtualenv(srcdir, datadir, preload_image, get_container_name):
    """
    Populate venv from preloaded image
    """
    try:
        if docker.is_boot2docker():
            docker.data_only_container(
                get_container_name('venv'),
                ['/usr/lib/ckan'],
                )
            img_id = docker.web_command(
                '/bin/mv /usr/lib/ckan/ /usr/lib/ckan_original',
                image=preload_image,
                commit=True,
                )
            docker.web_command(
                command='/bin/cp -a /usr/lib/ckan_original/. /usr/lib/ckan/.',
                volumes_from=get_container_name('venv'),
                image=img_id,
                )
            docker.remove_image(img_id)
            return

        docker.web_command(
            command='/bin/cp -a /usr/lib/ckan/. /usr/lib/ckan_target/.',
            rw={datadir + '/venv': '/usr/lib/ckan_target'},
            image=preload_image,
            )
    finally:
        rw = {datadir + '/venv': '/usr/lib/ckan'} if not docker.is_boot2docker() else {}
        volumes_from = get_container_name('venv') if docker.is_boot2docker() else None
        # fix venv permissions
        docker.web_command(
            command='/bin/chown -R --reference=/project /usr/lib/ckan',
            rw=rw,
            volumes_from=volumes_from,
            ro={srcdir: '/project'},
            )