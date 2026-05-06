def std_human_uid(kind=None):
    """Return a random generated human-friendly phrase as low-probability unique id"""

    kind_list = alphabet

    if kind == 'animal':
        kind_list = animals
    elif kind == 'place':
        kind_list = places

    name = "{color} {adjective} {kind} of {attribute}".format(
        color=choice(colors),
        adjective=choice(adjectives),
        kind=choice(kind_list),
        attribute=choice(attributes)
    )

    return name