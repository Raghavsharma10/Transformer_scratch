def full_award_group_funding_source(tag):
    """
    Given a funding group element
    Find the award group funding sources, one for each
    item found in the get_funding_group section
    """
    award_group_funding_sources = []
    funding_source_nodes = extract_nodes(tag, "funding-source")
    for funding_source_node in funding_source_nodes:

        award_group_funding_source = {}

        institution_nodes = extract_nodes(funding_source_node, 'institution')

        institution_node = first(institution_nodes)
        if institution_node:
            award_group_funding_source['institution'] = node_text(institution_node)
            if 'content-type' in institution_node.attrs:
                award_group_funding_source['institution-type'] = institution_node['content-type']

        institution_id_nodes = extract_nodes(funding_source_node, 'institution-id')
        institution_id_node = first(institution_id_nodes)
        if institution_id_node:
            award_group_funding_source['institution-id'] = node_text(institution_id_node)
            if 'institution-id-type' in institution_id_node.attrs:
                award_group_funding_source['institution-id-type'] = institution_id_node['institution-id-type']

        award_group_funding_sources.append(award_group_funding_source)

    return award_group_funding_sources