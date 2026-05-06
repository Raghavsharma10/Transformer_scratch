def clear():
    """
    Clear all data on the local server. Useful for debugging purposed.
    """
    utils.check_for_local_server()
    click.confirm(
        "Are you sure you want to do this? It will delete all of your data",
        abort=True
    )
    server = Server(config["local_server"]["url"])
    for db_name in all_dbs:
        del server[db_name]