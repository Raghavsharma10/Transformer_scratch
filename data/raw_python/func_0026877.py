def flatten_list(l: List[list]) -> list:
    """ takes a list of lists, l and returns a flat list
    """
    return [v for inner_l in l for v in inner_l]