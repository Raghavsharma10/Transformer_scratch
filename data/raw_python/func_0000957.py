def authors_json(soup):
    """authors list in article json format"""
    authors_json_data = []
    contributors_data = contributors(soup, "full")
    author_contributions_data = author_contributions(soup, None)
    author_competing_interests_data = competing_interests(soup, None)
    author_correspondence_data = full_correspondence(soup)
    authors_non_byline_data = authors_non_byline(soup)
    equal_contributions_map = map_equal_contributions(contributors_data)
    present_address_data = present_addresses(soup)
    foot_notes_data = other_foot_notes(soup)

    # First line authors builds basic structure
    for contributor in contributors_data:
        author_json = None
        if contributor["type"] == "author" and contributor.get("collab"):
            author_json = author_group(contributor, author_contributions_data,
                                       author_correspondence_data, author_competing_interests_data,
                                       equal_contributions_map, present_address_data,
                                       foot_notes_data)
        elif contributor.get("on-behalf-of"):
            author_json = author_on_behalf_of(contributor)
        elif contributor["type"] == "author" and not contributor.get("group-author-key"):
            author_json = author_person(contributor, author_contributions_data,
                                        author_correspondence_data, author_competing_interests_data,
                                        equal_contributions_map, present_address_data, foot_notes_data)

        if author_json:
            authors_json_data.append(author_json)

    # Second, add byline author data
    collab_map = collab_to_group_author_key_map(contributors_data)
    for contributor in [elem for elem in contributors_data if elem.get("group-author-key") and not elem.get("collab")]:
        for group_author in [elem for elem in authors_json_data if elem.get('type') == 'group']:
            group_author_key = None
            if group_author["name"] in collab_map:
                group_author_key = collab_map[group_author["name"]]
            if contributor.get("group-author-key") == group_author_key:
                author_json = author_person(contributor, author_contributions_data,
                                            author_correspondence_data, author_competing_interests_data,
                                            equal_contributions_map, present_address_data, foot_notes_data)
                if contributor.get("sub-group"):
                    if "groups" not in group_author:
                        group_author["groups"] = OrderedDict()
                    if contributor.get("sub-group") not in group_author["groups"]:
                        group_author["groups"][contributor.get("sub-group")] = []
                    group_author["groups"][contributor.get("sub-group")].append(author_json)
                else:
                    if "people" not in group_author:
                        group_author["people"] = []
                    group_author["people"].append(author_json)

    authors_json_data_rewritten = elifetools.json_rewrite.rewrite_json("authors_json", soup, authors_json_data)
    return authors_json_data_rewritten