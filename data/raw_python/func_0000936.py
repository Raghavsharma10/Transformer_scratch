def award_group_principal_award_recipient(tag):
    """
    Find the award group principal award recipient, one for each
    item found in the get_funding_group section
    """
    award_group_principal_award_recipient = []
    principal_award_recipients = extract_nodes(tag, "principal-award-recipient")

    for t in principal_award_recipients:
        principal_award_recipient_text = ""

        institution = node_text(first(extract_nodes(t, "institution")))
        surname = node_text(first(extract_nodes(t, "surname")))
        given_names = node_text(first(extract_nodes(t, "given-names")))
        string_name = node_text(first(raw_parser.string_name(t)))
        # Concatenate name and institution values if found
        #  while filtering out excess whitespace
        if(given_names):
            principal_award_recipient_text += given_names
        if(principal_award_recipient_text != ""):
            principal_award_recipient_text += " "
        if(surname):
            principal_award_recipient_text += surname
        if(institution):
            principal_award_recipient_text += institution
        if(string_name):
            principal_award_recipient_text += string_name

        award_group_principal_award_recipient.append(principal_award_recipient_text)
    return award_group_principal_award_recipient