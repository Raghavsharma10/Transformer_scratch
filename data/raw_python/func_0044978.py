def update(self, **kwargs):
        """
        Updates the matching objects for specified fields.

        Note:
            Post/pre save hooks and signals will NOT triggered.

            Unlike RDBMS systems, this method makes individual save calls
            to backend DB store. So this is exists as more of a comfortable
            utility method and not a performance enhancement.

        Keyword Args:
            \*\*kwargs: Fields with their corresponding values to be updated.

        Returns:
            Int. Number of updated objects.

        Example:
            .. code-block:: python

                Entry.objects.filter(pub_date__lte=2014).update(comments_on=False)
        """
        do_simple_update = kwargs.get('simple_update', True)
        no_of_updates = 0
        for model in self:
            no_of_updates += 1
            model._load_data(kwargs)
            model.save(internal=True)
        return no_of_updates