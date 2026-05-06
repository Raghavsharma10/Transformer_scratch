def translate_section(data):
    """ Translates data where data["Type"]=="Section" """
    sect_str = ""
    elements = data.get("Elements", [])
    for elem in elements:
            print("    Translating " + elem["Type"])
            sect_str += translate_map[elem["Type"]](elem)
    return sect_str