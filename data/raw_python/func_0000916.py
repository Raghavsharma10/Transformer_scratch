def supplementary_material(soup):
    """
    supplementary-material tags
    """
    supplementary_material = []

    supplementary_material_tags = raw_parser.supplementary_material(soup)

    position = 1

    for tag in supplementary_material_tags:
        item = {}

        copy_attribute(tag.attrs, 'id', item)

        # Get the tag type
        nodenames = ["supplementary-material"]
        details = tag_details(tag, nodenames)
        copy_attribute(details, 'type', item)
        copy_attribute(details, 'asset', item)
        copy_attribute(details, 'component_doi', item)
        copy_attribute(details, 'sibling_ordinal', item)

        if raw_parser.label(tag):
            item['label'] = node_text(raw_parser.label(tag))
            item['full_label'] = node_contents_str(raw_parser.label(tag))

        # Increment the position
        item['position'] = position
        # Ordinal should be the same as position in this case but set it anyway
        item['ordinal'] = tag_ordinal(tag)

        supplementary_material.append(item)

    return supplementary_material