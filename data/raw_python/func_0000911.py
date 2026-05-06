def tag_details(tag, nodenames):
    """
    Used in media and graphics to extract data from their parent tags
    """
    details = {}

    details['type'] = tag.name
    details['ordinal'] = tag_ordinal(tag)

    # Ordinal value
    if tag_details_sibling_ordinal(tag):
        details['sibling_ordinal'] = tag_details_sibling_ordinal(tag)

    # Asset name
    if tag_details_asset(tag):
        details['asset'] = tag_details_asset(tag)

    object_id_tag = first(raw_parser.object_id(tag, pub_id_type= "doi"))
    if object_id_tag:
        details['component_doi'] = extract_component_doi(tag, nodenames)

    return details