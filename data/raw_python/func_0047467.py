def update_module_types():
    """
    Download the repositories for all of the firmware_module_type records and
    update them using the `module.json` files from the repositories themselves.
    Currently only works for git repositories.
    """
    local_url = config["local_server"]["url"]
    server = Server(local_url)
    db = server[FIRMWARE_MODULE_TYPE]
    temp_folder = mkdtemp()
    for _id in db:
        if _id.startswith("_"):
            continue
        obj = db[_id]
        new_obj = update_record(FirmwareModuleType(obj), temp_folder)
        new_obj["_rev"] = obj["_rev"]
        if new_obj != obj:
            db[_id] = new_obj
    rmtree(temp_folder)