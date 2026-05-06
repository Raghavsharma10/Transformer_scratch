def cli_char(name, tibiadata, json):
    """Displays information about a Tibia character."""
    name = " ".join(name)
    char = _fetch_and_parse(Character.get_url, Character.from_content,
                            Character.get_url_tibiadata, Character.from_tibiadata,
                            tibiadata, name)
    if json and char:
        print(char.to_json(indent=2))
        return
    print(get_character_string(char))