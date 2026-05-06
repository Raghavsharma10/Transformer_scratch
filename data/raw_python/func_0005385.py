def setup_ci():
    # type: () -> None
    """ Setup AppEngine SDK on CircleCI """
    gcloud_path = shell.run('which gcloud', capture=True).stdout.strip()
    sdk_path = normpath(join(gcloud_path, '../../platform/google_appengine'))
    gcloud_cmd = gcloud_path + ' --quiet'

    if not exists(sdk_path):
        log.info("Installing AppEngine SDK")
        shell.run('sudo {} components install app-engine-python'.format(
            gcloud_cmd
        ))
    else:
        # Only initialise once. To reinitialise, just build without cache.
        log.info("AppEngine SDK already initialised")

    log.info("Using service account authentication")
    shell.run('{} auth activate-service-account --key-file {}'.format(
        gcloud_cmd,
        conf.proj_path('ops/client_secret.json')
    ))