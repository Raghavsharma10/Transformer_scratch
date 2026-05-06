def component_doi(soup):
    """
    Look for all object-id of pub-type-id = doi, these are the component DOI tags
    """
    component_doi = []

    object_id_tags = raw_parser.object_id(soup, pub_id_type = "doi")

    # Get components too for later
    component_list = components(soup)

    position = 1

    for tag in object_id_tags:
        component_object = {}
        component_object["doi"] = doi_uri_to_doi(tag.text)
        component_object["position"] = position

        # Try to find the type of component
        for component in component_list:
            if "doi" in component and component["doi"] == component_object["doi"]:
                component_object["type"] = component["type"]

        component_doi.append(component_object)

        position = position + 1

    return component_doi