def list_inventory(inventory):
    """List a projects inventory

    Given a project, simply list the contents of `inventory.yaml`

    Arguments:
        root (str): Absolute path to the `be` root directory,
            typically the current working directory.
        inventory (dict): inventory.yaml

    """

    inverted = invert_inventory(inventory)
    items = list()
    for item in sorted(inverted, key=lambda a: (inverted[a], a)):
        items.append((item, inverted[item]))
    return items