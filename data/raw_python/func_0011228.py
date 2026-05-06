def invert_inventory(inventory):
    """Return {item: binding} from {binding: item}

    Protect against items with additional metadata
    and items whose type is a number

    Returns:
        Dictionary of inverted inventory

    """

    inverted = dict()
    for binding, items in inventory.iteritems():
        for item in items:
            if isinstance(item, dict):
                item = item.keys()[0]
            item = str(item)  # Key may be number

            if item in inverted:
                echo("Warning: Duplicate item found, "
                     "for \"%s: %s\"" % (binding, item))
                continue
            inverted[item] = binding

    return inverted