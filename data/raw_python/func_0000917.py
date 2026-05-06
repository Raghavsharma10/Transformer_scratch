def contrib_email(contrib_tag):
    """
    Given a contrib tag, look for an email tag, and
    only return the value if it is not inside an aff tag
    """
    email = []
    for email_tag in extract_nodes(contrib_tag, "email"):
        if email_tag.parent.name != "aff":
            email.append(email_tag.text)
    return email if len(email) > 0 else None