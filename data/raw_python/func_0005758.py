def _add_new_next_method(cls):
    """
    TODO
    """

    def new_next(self):
        field_values = [next(g) for g in self.field_gens.values()]
        return self.item_cls(*field_values)

    cls.__next__ = new_next