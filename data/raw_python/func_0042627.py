def cli_guilds(world, tibiadata, json):
    """Displays the list of guilds for a specific world"""
    world = " ".join(world)
    guilds = _fetch_and_parse(ListedGuild.get_world_list_url, ListedGuild.list_from_content,
                              ListedGuild.get_world_list_url_tibiadata, ListedGuild.list_from_tibiadata,
                              tibiadata, world)
    if json and guilds:
        import json as _json
        print(_json.dumps(guilds, default=dict, indent=2))
        return
    print(get_guilds_string(guilds))