def deinit(ctx):
    """
    Detach from the current cloud server
    """
    utils.check_for_cloud_server()
    if config["local_server"]["url"]:
        utils.cancel_global_db_replication()
    if config["cloud_server"]["username"]:
        ctx.invoke(logout_user)
    config["cloud_server"]["url"] = None