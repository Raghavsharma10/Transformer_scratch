def __initial_minus_queryset(self):
        """
        Gives all elements from self._initial having a slot value that is not already
        in self.get_queryset()
        """
        queryset = self.get_queryset()

        def initial_not_in_queryset(initial):
            for x in queryset:
                if x.slot == initial['slot']:
                    return False

            return True

        return list(filter(initial_not_in_queryset, self._initial))