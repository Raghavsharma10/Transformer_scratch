def media(soup):
    """
    All media tags and some associated data about the related component doi
    and the parent of that doi (not always present)
    """
    media = []

    media_tags = raw_parser.media(soup)

    position = 1

    for tag in media_tags:
        media_item = {}

        copy_attribute(tag.attrs, 'mime-subtype', media_item)
        copy_attribute(tag.attrs, 'mimetype', media_item)
        copy_attribute(tag.attrs, 'xlink:href', media_item, 'xlink_href')
        copy_attribute(tag.attrs, 'content-type', media_item)

        nodenames = ["sub-article", "media", "fig-group", "fig", "supplementary-material"]

        details = tag_details(tag, nodenames)
        copy_attribute(details, 'component_doi', media_item)
        copy_attribute(details, 'type', media_item)
        copy_attribute(details, 'sibling_ordinal', media_item)

        # Try to get the component DOI of the parent tag
        parent_tag = first_parent(tag, nodenames)
        if parent_tag:
            acting_parent_tag = component_acting_parent_tag(parent_tag, tag)
            if acting_parent_tag:
                details = tag_details(acting_parent_tag, nodenames)
                copy_attribute(details, 'type', media_item, 'parent_type')
                copy_attribute(details, 'ordinal', media_item, 'parent_ordinal')
                copy_attribute(details, 'asset', media_item, 'parent_asset')
                copy_attribute(details, 'sibling_ordinal', media_item, 'parent_sibling_ordinal')
                copy_attribute(details, 'component_doi', media_item, 'parent_component_doi')

            # Try to get the parent parent
            p_parent_tag = first_parent(parent_tag, nodenames)
            if p_parent_tag:
                acting_p_parent_tag = component_acting_parent_tag(p_parent_tag, parent_tag)
                if acting_p_parent_tag:
                    details = tag_details(acting_p_parent_tag, nodenames)
                    copy_attribute(details, 'type', media_item, 'p_parent_type')
                    copy_attribute(details, 'ordinal', media_item, 'p_parent_ordinal')
                    copy_attribute(details, 'asset', media_item, 'p_parent_asset')
                    copy_attribute(details, 'sibling_ordinal', media_item, 'p_parent_sibling_ordinal')
                    copy_attribute(details, 'component_doi', media_item, 'p_parent_component_doi')

                # Try to get the parent parent parent
                p_p_parent_tag = first_parent(p_parent_tag, nodenames)
                if p_p_parent_tag:
                    acting_p_p_parent_tag = component_acting_parent_tag(p_p_parent_tag, p_parent_tag)
                    if acting_p_p_parent_tag:
                        details = tag_details(acting_p_p_parent_tag, nodenames)
                        copy_attribute(details, 'type', media_item, 'p_p_parent_type')
                        copy_attribute(details, 'ordinal', media_item, 'p_p_parent_ordinal')
                        copy_attribute(details, 'asset', media_item, 'p_p_parent_asset')
                        copy_attribute(details, 'sibling_ordinal', media_item, 'p_p_parent_sibling_ordinal')
                        copy_attribute(details, 'component_doi', media_item, 'p_p_parent_component_doi')

        # Increment the position
        media_item['position'] = position
        # Ordinal should be the same as position in this case but set it anyway
        media_item['ordinal'] = tag_ordinal(tag)

        media.append(media_item)

        position += 1

    return media