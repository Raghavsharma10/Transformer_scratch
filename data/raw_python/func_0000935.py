def award_group_award_id(tag):
    """
    Find the award group award id, one for each
    item found in the get_funding_group section
    """
    award_group_award_id = []
    award_id_tags = extract_nodes(tag, "award-id")
    for t in award_id_tags:
        award_group_award_id.append(t.text)
    return award_group_award_id