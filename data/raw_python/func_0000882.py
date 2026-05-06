def person_same_name_map(json_content, role_from):
    "to merge multiple editors into one record, filter by role values and group by name"
    matched_editors = [(i, person) for i, person in enumerate(json_content)
                       if person.get('role') in role_from]
    same_name_map = {}
    for i, editor in matched_editors:
        if not editor.get("name"):
            continue
        # compare name of each
        name = editor.get("name").get("index")
        if name not in same_name_map:
            same_name_map[name] = []
        same_name_map[name].append(i)
    return same_name_map