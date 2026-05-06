def _linearize(interface):
    """
    Return a list of all the bases of a given interface in depth-first order.

    @param interface: an Interface object.

    @return: a L{list} of Interface objects, the input in all its bases, in
    subclass-to-base-class, depth-first order.
    """
    L = [interface]
    for baseInterface in interface.__bases__:
        if baseInterface is not Interface:
            L.extend(_linearize(baseInterface))
    return L