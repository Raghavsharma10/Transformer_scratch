def _build_category_tree(slug, reference=None, items=None):
    """
    Builds a recursive tree with category relations as children.
    """

    if items is None:
        items = []

    for key in reference:
        category = reference[key]

        if category["parent"] == slug:
            children = _build_category_tree(category["nicename"],
                                            reference=reference)
            category["children"] = children
            items.append(category)

    return items