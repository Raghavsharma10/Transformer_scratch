def superdict(arg=()):
    """Recursive defaultdict which can init with other dict """
    def update(obj, arg):
        return obj.update(arg) or obj
    return update(defaultdict(superdict), arg)