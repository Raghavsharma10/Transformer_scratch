def load_fixture(fixture_file):
    """
    Populate the database from a JSON file. Reads the JSON file FIXTURE_FILE
    and uses it to populate the database. Fuxture files should consist of a
    dictionary mapping database names to arrays of objects to store in those
    databases.
    """
    utils.check_for_local_server()
    local_url = config["local_server"]["url"]
    server = Server(local_url)
    fixture = json.load(fixture_file)
    for db_name, _items in fixture.items():
        db = server[db_name]
        with click.progressbar(
            _items, label=db_name, length=len(_items)
        ) as items:
            for item in items:
                item_id = item["_id"]
                if item_id in db:
                    old_item = db[item_id]
                    item["_rev"] = old_item["_rev"]
                    if item == old_item:
                        continue
                db[item_id] = item