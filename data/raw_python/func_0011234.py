def binding_from_item(inventory, item):
    """Return binding for `item`

    Example:
        asset:
        - myasset

        The binding is "asset"

    Arguments:
        project: Name of project
        item (str): Name of item

    """

    if item in self.bindings:
        return self.bindings[item]

    bindings = invert_inventory(inventory)

    try:
        self.bindings[item] = bindings[item]
        return bindings[item]

    except KeyError as exc:
        exc.bindings = bindings
        raise exc