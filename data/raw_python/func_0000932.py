def award_groups(soup):
    """
    Find the award-group items and return a list of details
    """
    award_groups = []

    funding_group_section = extract_nodes(soup, "funding-group")
    for fg in funding_group_section:

        award_group_tags = extract_nodes(fg, "award-group")

        for ag in award_group_tags:

            award_group = {}

            award_group['funding_source'] = award_group_funding_source(ag)
            award_group['recipient'] = award_group_principal_award_recipient(ag)
            award_group['award_id'] = award_group_award_id(ag)

            award_groups.append(award_group)

    return award_groups