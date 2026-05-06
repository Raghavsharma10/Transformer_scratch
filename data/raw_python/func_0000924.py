def extract_component_doi(tag, nodenames):
    """
    Used to get component DOI from a tag and confirm it is actually for that tag
    and it is not for one of its children in the list of nodenames
    """
    component_doi = None

    if(tag.name == "sub-article"):
        component_doi = doi_uri_to_doi(node_text(first(raw_parser.article_id(tag, pub_id_type= "doi"))))
    else:
        object_id_tag = first(raw_parser.object_id(tag, pub_id_type= "doi"))
        # Tweak: if it is media and has no object_id_tag then it is not a "component"
        if tag.name == "media" and not object_id_tag:
            component_doi = None
        else:
            # Check the object id is for this tag and not one of its children
            #   This happens for example when boxed text has a child figure,
            #   the boxed text does not have a DOI, the figure does have one
            if object_id_tag and first_parent(object_id_tag, nodenames).name == tag.name:
                component_doi = doi_uri_to_doi(node_text(object_id_tag))

    return component_doi