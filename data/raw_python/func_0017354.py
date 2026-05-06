def get_custom_implementations(self):
        """Retrieve a list of cutom implementations.

        Yields:
            (str, str, ImplementationProperty) tuples: The name of the attribute
                an implementation lives at, the name of the related transition,
                and the related implementation.
        """
        for trname in self.custom_implems:
            attr = self.transitions_at[trname]
            implem = self.implementations[trname]
            yield (trname, attr, implem)