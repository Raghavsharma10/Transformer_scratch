def create_source(srcdir, preload_image, datapusher=False):
    """
    Copy ckan source, datapusher source (optional), who.ini and schema.xml
    from preload image into srcdir
    """
    try:
        docker.web_command(
            command='/bin/cp -a /project/ckan /project_target/ckan',
            rw={srcdir: '/project_target'},
            image=preload_image)
        if datapusher:
            docker.web_command(
                command='/bin/cp -a /project/datapusher /project_target/datapusher',
                rw={srcdir: '/project_target'},
                image=preload_image)
        shutil.copy(
            srcdir + '/ckan/ckan/config/who.ini',
            srcdir)
        shutil.copy(
            srcdir + '/ckan/ckan/config/solr/schema.xml',
            srcdir)
    finally:
        # fix srcdir permissions
        docker.web_command(
            command='/bin/chown -R --reference=/project /project',
            rw={srcdir: '/project'},
            )