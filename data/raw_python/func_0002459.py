def new_pivot(self, attributes=None, exists=False):
        """
        Create a new pivot model instance.
        """
        from .morph_pivot import MorphPivot

        pivot = MorphPivot(self._parent, attributes, self._table, exists)

        pivot.set_pivot_keys(self._foreign_key, self._other_key)\
            .set_morph_type(self._morph_type)\
            .set_morph_class(self._morph_class)

        return pivot