def add_new_next_method(obj):
    """
    TODO
    """

    def new_next(self):
        field_values = [next(g) for g in self.field_gens.values()]
        return self.item_cls(*field_values)

    obj.__next__ = new_next