def logout(ctx):
    """ Log out of your user account """
    check_for_cloud_server()
    check_for_cloud_user()
    if config["cloud_server"]["farm_name"]:
        ctx.invoke(deinit_farm)
    config["cloud_server"]["username"] = None
    config["cloud_server"]["password"] = None