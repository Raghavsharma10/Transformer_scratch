def _parse_tags(element):
    """
    Retrieves and parses tags into a array/dict.

    Example:

        [{"term_id": 1, "slug": "python", "name": "Python"},
        {"term_id": 2, "slug": "java", "name": "Java"}]
    """

    tags = []
    items = element.findall("./{%s}tag" % WP_NAMESPACE)

    for item in items:
        term_id = item.find("./{%s}term_id" % WP_NAMESPACE).text
        slug = item.find("./{%s}tag_slug" % WP_NAMESPACE).text
        name = item.find("./{%s}tag_name" % WP_NAMESPACE).text

        tag = {
            "term_id": term_id,
            "slug": slug,
            "name": name,
        }

        tags.append(tag)

    return tags