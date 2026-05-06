def full_award_groups(soup):
    """
    Find the award-group items and return a list of details
    """
    award_groups = []

    funding_group_section = extract_nodes(soup, "funding-group")
    # counter for auto generated id values, if required
    generated_id_counter = 1
    for fg in funding_group_section:

        award_group_tags = extract_nodes(fg, "award-group")

        for ag in award_group_tags:
            if 'id' in ag.attrs:
                ref = ag['id']
            else:
                # hack: generate and increment an id value none is available
                ref = "award-group-{id}".format(id=generated_id_counter)
                generated_id_counter += 1

            award_group = {}
            award_group_id = award_group_award_id(ag)
            if award_group_id is not None:
                award_group['award-id'] = first(award_group_id)
            funding_sources = full_award_group_funding_source(ag)
            source = first(funding_sources)
            if source is not None:
                copy_attribute(source, 'institution', award_group)
                copy_attribute(source, 'institution-id', award_group, 'id')
                copy_attribute(source, 'institution-id-type', award_group, destination_key='id-type')
            award_group_by_ref = {}
            award_group_by_ref[ref] = award_group
            award_groups.append(award_group_by_ref)

    return award_groups