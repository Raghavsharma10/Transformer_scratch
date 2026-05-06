def get(self, key=None, **kwargs):
        """
        Ensures that only one result is returned from DB and raises an exception otherwise.
        Can work in 3 different way.

            - If no argument is given, only does "ensuring about one and only object" job.
            - If key given as only argument, retrieves the object from DB.
            - if query filters given, implicitly calls filter() method.

        Raises:
            MultipleObjectsReturned: If there is more than one (1) record is returned.
        """
        clone = copy.deepcopy(self)
        # If we are in a slice, adjust the start and rows
        if self._start:
            clone.adapter.set_params(start=self._start)
        if self._rows:
            clone.adapter.set_params(rows=self._rows)
        if key:
            data, key = clone.adapter.get(key)
        elif kwargs:
            data, key = clone.filter(**kwargs).adapter.get()
        else:
            data, key = clone.adapter.get()
        if clone._cfg['rtype'] == ReturnType.Object:
            return data, key
        return self._make_model(data, key)