def _parse_authors(element):
    """
    Returns a well formatted list of users that can be matched against posts.
    """

    authors = []
    items = element.findall("./{%s}author" % WP_NAMESPACE)

    for item in items:
        login = item.find("./{%s}author_login" % WP_NAMESPACE).text
        email = item.find("./{%s}author_email" % WP_NAMESPACE).text
        first_name = item.find("./{%s}author_first_name" % WP_NAMESPACE).text
        last_name = item.find("./{%s}author_last_name" % WP_NAMESPACE).text
        display_name = item.find(
            "./{%s}author_display_name" % WP_NAMESPACE).text

        authors.append({
            "login": login,
            "email": email,
            "display_name": display_name,
            "first_name": first_name,
            "last_name": last_name
        })

    return authors