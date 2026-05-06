def load_parent_implems(self, parent_implems):
        """Import previously defined implementations.

        Args:
            parent_implems (ImplementationList): List of implementations defined
                in a parent class.
        """
        for trname, attr, implem in parent_implems.get_custom_implementations():
            self.implementations[trname] = implem.copy()
            self.transitions_at[trname] = attr
            self.custom_implems.add(trname)