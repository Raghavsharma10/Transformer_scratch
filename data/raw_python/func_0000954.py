def author_json_details(author, author_json, contributions, correspondence,
                        competing_interests, equal_contributions_map, present_address_data,
                        foot_notes_data, html_flag=True):
    # Configure the XML to HTML conversion preference for shorthand use below
    convert = lambda xml_string: xml_to_html(html_flag, xml_string)

    """add more author json"""
    if author_affiliations(author):
        author_json["affiliations"] = author_affiliations(author)

    # foot notes or additionalInformation
    if author_foot_notes(author, foot_notes_data):
        author_json["additionalInformation"] = author_foot_notes(author, foot_notes_data)

    # email
    if author_email_addresses(author, correspondence):
        author_json["emailAddresses"] = author_email_addresses(author, correspondence)

    # phone
    if author_phone_numbers(author, correspondence):
        author_json["phoneNumbers"] = author_phone_numbers_json(author, correspondence)

    # contributions
    if author_contribution(author, contributions):
        author_json["contribution"] = convert(author_contribution(author, contributions))

    # competing interests
    if author_competing_interests(author, competing_interests):
        author_json["competingInterests"] = convert(
            author_competing_interests(author, competing_interests))

    # equal-contributions
    if author_equal_contribution(author, equal_contributions_map):
        author_json["equalContributionGroups"] = author_equal_contribution(author, equal_contributions_map)

    # postalAddress
    if author_present_address(author, present_address_data):
        author_json["postalAddresses"] = author_present_address(author, present_address_data)

    return author_json