def translate_tab(data):
    """ Translates data where data["Type"]=="Tab" """
    tab_str = ""
    sections = data.get("Sections", [])
    for section in sections:
            print("  Translating " + section["Type"])
            tab_str += translate_map[section["Type"]](section)
    return tab_str