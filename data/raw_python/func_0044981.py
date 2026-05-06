def values_list(self, *args, **kwargs):
        """
        Returns list of values for given fields.
        Since this will implicitly use data() method,
        it's more efficient than simply looping through model instances.


        Args:
            flatten (bool): True. Flatten if there is only one field name given.
             Returns ['one','two', 'three'] instead of
             [['one'], ['two'], ['three]]
            \*args: List of fields to be retured as list.

        Returns:
            List of deleted objects or None if *confirm* not set.

        Example:
            >>> Person.objects.filter(age__gte=16).values_list('name', 'lastname')

        """
        results = []
        for data, key in self.data():
            results.append([data[val] if val != 'key' else key for val in args])
        return results if len(args) > 1 or not kwargs.get('flatten', True) else [
            i[0] for i in results]