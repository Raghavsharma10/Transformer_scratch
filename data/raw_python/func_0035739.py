def save(self):
        """Save this entry.

        If the entry does not have an :attr:`id`, a new id will be assigned,
        and the :attr:`id` attribute set accordingly.

        Pre-save processing of the fields saved can be done by
        overriding the :meth:`prepare_save` method.

        Additional actions to be done after the save operation
        has been completed can be added by defining the
        :meth:`post_save` method.

        """
        id = self.id or self.objects.id(self.name)
        self.objects[id] = self.prepare_save(dict(self))
        self.id = id
        self.post_save()
        return id