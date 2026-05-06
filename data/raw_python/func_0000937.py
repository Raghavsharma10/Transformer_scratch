def object_id_doi(tag, parent_tag_name=None):
    """DOI in an object-id tag found inside the tag"""
    doi = None
    object_id = None
    object_ids = raw_parser.object_id(tag, "doi")
    if object_ids:
        object_id = first([id_ for id_ in object_ids])
    if parent_tag_name and object_id and object_id.parent.name != parent_tag_name:
        object_id = None
    if object_id:
        doi = node_contents_str(object_id)
    return doi