def filter(self, all_records=False, **filters):
        """
        Applies given query filters. If wanted result is more than specified size,
        exception is raised about using all() method instead of filter.

        Args:
            all_records (bool):
            **filters: Query filters as keyword arguments.

        Returns:
            Self. Queryset object.

        Examples:
            >>> Person.objects.filter(name='John') # same as .filter(name__exact='John')
            >>> Person.objects.filter(age__gte=16, name__startswith='jo')
            >>> # Assume u1 and u2 as related model instances.
            >>> Person.objects.filter(work_unit__in=[u1, u2], name__startswith='jo')
        """

        clone = copy.deepcopy(self)
        clone.adapter.add_query(filters.items())
        clone_length = clone.count()
        if clone_length > self._cfg['row_size'] and not all_records:
            raise Exception("""Your query result count(%s) is more than specified result value(%s).
            You can narrow your filters, you can apply your own pagination or
            you can use all() method for getting all filter results.
            Example Usage: Unit.objects.all()
            
            Filters: %s  Model Class: %s 
            """ % (clone_length, self._cfg['row_size'], filters, self._cfg['model_class']))

        return clone