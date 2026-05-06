def deinit_farm():
    """
    Detach from the current farm. Cancels the replication between your local
    server and the cloud instance if it is set up.
    """
    utils.check_for_cloud_server()
    utils.check_for_cloud_user()
    utils.check_for_cloud_farm()
    farm_name = config["cloud_server"]["farm_name"]
    if farm_name and config["local_server"]["url"]:
        utils.cancel_per_farm_db_replication()
    config["cloud_server"]["farm_name"] = None