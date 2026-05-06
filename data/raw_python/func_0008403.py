def product(*args, **kwargs):
    """ Yields all permutations with replacement:
        list(product("cat", repeat=2)) => 
        [("c", "c"), 
         ("c", "a"), 
         ("c", "t"), 
         ("a", "c"), 
         ("a", "a"), 
         ("a", "t"), 
         ("t", "c"), 
         ("t", "a"), 
         ("t", "t")]
    """
    p = [[]]
    for iterable in map(tuple, args) * kwargs.get("repeat", 1):
        p = [x + [y] for x in p for y in iterable]
    for p in p:
        yield tuple(p)