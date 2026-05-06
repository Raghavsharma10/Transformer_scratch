def components(soup):
    """
    Find the components, i.e. those parts that would be assigned
    a unique component DOI, such as figures, tables, etc.
    - position is in what order the tag appears in the entire set of nodes
    - ordinal is in what order it is for all the tags of its own type
    """
    components = []

    nodenames = ["abstract", "fig", "table-wrap", "media",
                 "chem-struct-wrap", "sub-article", "supplementary-material",
                 "boxed-text", "app"]

    # Count node order overall
    position = 1

    position_by_type = {}
    for nodename in nodenames:
        position_by_type[nodename] = 1

    article_doi = doi(soup)

    # Find all tags for all component_types, allows the order
    #  in which they are found to be preserved
    component_tags = extract_nodes(soup, nodenames)

    for tag in component_tags:

        component = OrderedDict()

        # Component type is the tag's name
        ctype = tag.name

        # First find the doi if present
        component_doi = extract_component_doi(tag, nodenames)
        if component_doi is None:
            continue
        else:
            component['doi'] = doi_uri_to_doi(component_doi)
            component['doi_url'] = doi_to_doi_uri(component['doi'])

        copy_attribute(tag.attrs, 'id', component)

        if(ctype == "sub-article"):
            title_tag = raw_parser.article_title(tag)
        elif(ctype == "boxed-text"):
            title_tag = title_tag_inspected(tag, tag.name, direct_sibling_only=True)
            if not title_tag:
                title_tag = title_tag_inspected(tag, "caption", "boxed-text")
            # New kitchen sink has boxed-text inside app tags, tag the sec tag title if so
            #  but do not take it if there is a caption
            if (not title_tag and tag.parent and tag.parent.name in ["sec", "app"]
                and not caption_tag_inspected(tag, tag.name)):
                title_tag = title_tag_inspected(tag.parent, tag.parent.name, direct_sibling_only=True)
        else:
            title_tag = raw_parser.title(tag)

        if title_tag:
            component['title'] = node_text(title_tag)
            component['full_title'] = node_contents_str(title_tag)

        if ctype == "boxed-text":
            label_tag = label_tag_inspected(tag, "boxed-text")
        else:
            label_tag = raw_parser.label(tag)

        if label_tag:
            component['label'] = node_text(label_tag)
            component['full_label'] = node_contents_str(label_tag)

        if raw_parser.caption(tag):
            first_paragraph = first(paragraphs(raw_parser.caption(tag)))
            # fix a problem with the new kitchen sink of caption within caption tag
            if first_paragraph:
                nested_caption = raw_parser.caption(first_paragraph)
                if nested_caption:
                    nested_paragraphs = paragraphs(nested_caption) 
                    first_paragraph = first(nested_paragraphs) or first_paragraph
            if first_paragraph and not starts_with_doi(first_paragraph):
                # Remove the supplementary tag from the paragraph if present
                if raw_parser.supplementary_material(first_paragraph):
                    first_paragraph = remove_tag_from_tag(first_paragraph, 'supplementary-material')
                if node_text(first_paragraph).strip():
                    component['caption'] = node_text(first_paragraph)
                    component['full_caption'] = node_contents_str(first_paragraph)

        if raw_parser.permissions(tag):

            component['permissions'] = []
            for permissions_tag in raw_parser.permissions(tag):
                permissions_item = {}
                if raw_parser.copyright_statement(permissions_tag):
                    permissions_item['copyright_statement'] = \
                        node_text(raw_parser.copyright_statement(permissions_tag))

                if raw_parser.copyright_year(permissions_tag):
                    permissions_item['copyright_year'] = \
                        node_text(raw_parser.copyright_year(permissions_tag))

                if raw_parser.copyright_holder(permissions_tag):
                    permissions_item['copyright_holder'] = \
                        node_text(raw_parser.copyright_holder(permissions_tag))

                if raw_parser.licence_p(permissions_tag):
                    permissions_item['license'] = \
                        node_text(first(raw_parser.licence_p(permissions_tag)))
                    permissions_item['full_license'] = \
                        node_contents_str(first(raw_parser.licence_p(permissions_tag)))

                component['permissions'].append(permissions_item)

        if raw_parser.contributors(tag):
            component['contributors'] = []
            for contributor_tag in raw_parser.contributors(tag):
                component['contributors'].append(format_contributor(contributor_tag, soup))

        # There are only some parent tags we care about for components
        #  and only check two levels of parentage
        parent_nodenames = ["sub-article", "fig-group", "fig", "boxed-text", "table-wrap", "app", "media"]
        parent_tag = first_parent(tag, parent_nodenames)

        if parent_tag:

            # For fig-group we actually want the first fig of the fig-group as the parent
            acting_parent_tag = component_acting_parent_tag(parent_tag, tag)

            # Only counts if the acting parent tag has a DOI
            if (acting_parent_tag and \
               extract_component_doi(acting_parent_tag, parent_nodenames) is not None):

                component['parent_type'] = acting_parent_tag.name
                component['parent_ordinal'] = tag_ordinal(acting_parent_tag)
                component['parent_sibling_ordinal'] = tag_details_sibling_ordinal(acting_parent_tag)
                component['parent_asset'] = tag_details_asset(acting_parent_tag)

            # Look for parent parent, if available
            parent_parent_tag = first_parent(parent_tag, parent_nodenames)

            if parent_parent_tag:

                acting_parent_tag = component_acting_parent_tag(parent_parent_tag, parent_tag)

                if (acting_parent_tag and \
                   extract_component_doi(acting_parent_tag, parent_nodenames) is not None):
                    component['parent_parent_type'] = acting_parent_tag.name
                    component['parent_parent_ordinal'] = tag_ordinal(acting_parent_tag)
                    component['parent_parent_sibling_ordinal'] = tag_details_sibling_ordinal(acting_parent_tag)
                    component['parent_parent_asset'] = tag_details_asset(acting_parent_tag)

        content = ""
        for p_tag in extract_nodes(tag, "p"):
            if content != "":
                # Add a space before each new paragraph for now
                content = content + " "
            content = content + node_text(p_tag)

        if(content != ""):
            component['content'] = content

        # mime type
        media_tag = None
        if(ctype == "media"):
            media_tag = tag
        elif(ctype == "supplementary-material"):
            media_tag = first(raw_parser.media(tag))
        if media_tag:
            component['mimetype'] = media_tag.get("mimetype")
            component['mime-subtype'] = media_tag.get("mime-subtype")

        if(len(component) > 0):
            component['article_doi'] = article_doi
            component['type'] = ctype
            component['position'] = position

            # Ordinal is based on all tags of the same type even if they have no DOI
            component['ordinal'] = tag_ordinal(tag)
            component['sibling_ordinal'] = tag_details_sibling_ordinal(tag)
            component['asset'] = tag_details_asset(tag)
            #component['ordinal'] = position_by_type[ctype]

            components.append(component)

            position += 1
            position_by_type[ctype] += 1


    return components