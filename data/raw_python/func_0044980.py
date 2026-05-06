def delete(self):
        """
        Deletes all objects that matches to the queryset.

        Note:
            Unlike RDBMS systems, this method makes individual save calls
            to backend DB store. So this is exists as more of a comfortable
            utility method and not a performance enhancement.

        Returns:
            List of deleted objects or None if *confirm* not set.

        Example:
            >>> Person.objects.filter(age__gte=16, name__startswith='jo').delete()

        """
        clone = copy.deepcopy(self)
        # clone.adapter.want_deleted = True
        return [item.delete() and item for item in clone]