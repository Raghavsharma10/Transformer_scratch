def _parse_categories(element):
    """
    Returns a list with categories with relations.
    """
    reference = {}
    items = element.findall("./{%s}category" % WP_NAMESPACE)

    for item in items:
        term_id = item.find("./{%s}term_id" % WP_NAMESPACE).text
        nicename = item.find("./{%s}category_nicename" % WP_NAMESPACE).text
        name = item.find("./{%s}cat_name" % WP_NAMESPACE).text
        parent = item.find("./{%s}category_parent" % WP_NAMESPACE).text

        category = {
            "term_id": term_id,
            "nicename": nicename,
            "name": name,
            "parent": parent
        }

        reference[nicename] = category

    return _build_category_tree(None, reference=reference)