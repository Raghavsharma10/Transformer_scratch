def apply(self, collection, ops, resource=None, **kwargs):
        """Filter given collection."""
        mfield = self.mfield or resource.meta.model._meta.fields.get(self.field.attribute)
        if mfield:
            collection = collection.where(*[op(mfield, val) for op, val in ops])
        return collection