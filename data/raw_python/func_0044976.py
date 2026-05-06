def get_or_create(self, defaults=None, **kwargs):
        """
        Looks up an object with the given kwargs, creating a new one if necessary.

        Args:
            defaults (dict): Used when we create a new object. Must map to fields
                of the model.
            \*\*kwargs: Used both for filtering and new object creation.

        Returns:
            A tuple of (object, created), where created is a boolean variable
            specifies whether the object was newly created or not.

        Example:
            In the following example, *code* and *name* fields are used to query the DB.

            .. code-block:: python

                obj, is_new = Permission.objects.get_or_create({'description': desc},
                                                                code=code, name=name)

            {description: desc} dict is just for new creations. If we can't find any
            records by filtering on *code* and *name*, then we create a new object by
            using all of the inputs.


        """
        try:
            return self.get(**kwargs), False
        except ObjectDoesNotExist:
            pass

        data = defaults or {}
        data.update(kwargs)
        return self._model_class(**data).blocking_save(), True