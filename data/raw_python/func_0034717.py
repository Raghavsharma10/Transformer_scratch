def _parse_comments(element):
    """
    Returns a list with comments.
    """

    comments = []
    items = element.findall("./{%s}comment" % WP_NAMESPACE)

    for item in items:
        comment_id = item.find("./{%s}comment_id" % WP_NAMESPACE).text
        author = item.find("./{%s}comment_author" % WP_NAMESPACE).text
        email = item.find("./{%s}comment_author_email" % WP_NAMESPACE).text
        author_url = item.find("./{%s}comment_author_url" % WP_NAMESPACE).text
        author_ip = item.find("./{%s}comment_author_IP" % WP_NAMESPACE).text
        date = item.find("./{%s}comment_date" % WP_NAMESPACE).text
        date_gmt = item.find("./{%s}comment_date_gmt" % WP_NAMESPACE).text
        content = item.find("./{%s}comment_content" % WP_NAMESPACE).text
        approved = item.find("./{%s}comment_approved" % WP_NAMESPACE).text
        comment_type = item.find("./{%s}comment_type" % WP_NAMESPACE).text
        parent = item.find("./{%s}comment_parent" % WP_NAMESPACE).text
        user_id = item.find("./{%s}comment_user_id" % WP_NAMESPACE).text

        comment = {
            "id": comment_id,
            "author": author,
            "author_email": email,
            "author_url": author_url,
            "author_ip": author_ip,
            "date": date,
            "date_gmt": date_gmt,
            "content": content,
            "approved": approved,
            "type": comment_type,
            "parent": parent,
            "user_id": user_id,
        }

        comments.append(comment)

    return comments