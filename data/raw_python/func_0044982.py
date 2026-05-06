def values(self, *args):
        """
        Returns list of dicts (field names as keys) for given fields.

        Args:
            \*args: List of fields to be returned as dict.

        Returns:
            list of dicts for given fields.

        Example:
            >>> Person.objects.filter(age__gte=16, name__startswith='jo').values('name', 'lastname')

        """
        return [dict(zip(args, values_list))
                for values_list in self.values_list(flatten=False, *args)]