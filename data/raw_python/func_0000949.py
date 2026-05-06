def body_block_attribution(tag):
    "extract the attribution content for figures, tables, videos"
    attributions = []
    if raw_parser.attrib(tag):
        for attrib_tag in raw_parser.attrib(tag):
            attributions.append(node_contents_str(attrib_tag))
    if raw_parser.permissions(tag):
        # concatenate content from from the permissions tag
        for permissions_tag in raw_parser.permissions(tag):
            attrib_string = ''
            # add the copyright statement if found
            attrib_string = join_sentences(attrib_string,
                node_contents_str(raw_parser.copyright_statement(permissions_tag)), '.')
            # add the license paragraphs
            if raw_parser.licence_p(permissions_tag):
                for licence_p_tag in raw_parser.licence_p(permissions_tag):
                    attrib_string = join_sentences(attrib_string,
                                                   node_contents_str(licence_p_tag), '.')
            if attrib_string != '':
                attributions.append(attrib_string)
    return attributions