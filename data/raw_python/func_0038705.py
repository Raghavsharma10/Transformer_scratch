def sort(self, *sorting, **kwargs):
        """Sort resources."""
        sorting_ = []
        for name, desc in sorting:
            field = self.meta.model._meta.fields.get(name)
            if field is None:
                continue
            if desc:
                field = field.desc()
            sorting_.append(field)
        if sorting_:
            return self.collection.order_by(*sorting_)
        return self.collection