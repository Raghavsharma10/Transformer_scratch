def award_group_funding_source(tag):
    """
    Given a funding group element
    Find the award group funding sources, one for each
    item found in the get_funding_group section
    """
    award_group_funding_source = []
    funding_source_tags = extract_nodes(tag, "funding-source")
    for t in funding_source_tags:
        award_group_funding_source.append(t.text)
    return award_group_funding_source