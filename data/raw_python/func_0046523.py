def init(cloud_url):
    """
    Choose a cloud server to use. Sets CLOUD_URL as the cloud server to use and
    sets up replication of global databases from that cloud server if a local
    database is already initialized (via `openag db init`).
    """
    old_cloud_url = config["cloud_server"]["url"]
    if old_cloud_url and old_cloud_url != cloud_url:
        raise click.ClickException(
            'Server "{}" already selected. Call `openag cloud deinit` to '
            'detach from that server before selecting a new one'.format(
                old_cloud_url
            )
        )
    parsed_url = urlparse(cloud_url)
    if not parsed_url.scheme or not parsed_url.netloc or not parsed_url.port:
        raise click.BadParameter("Invalid url")
    if config["local_server"]["url"]:
        utils.replicate_global_dbs(cloud_url=cloud_url)
    config["cloud_server"]["url"] = cloud_url