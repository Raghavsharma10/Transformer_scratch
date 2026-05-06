def fill_attrs(self, attrs):
        """Update the 'attrs' dict with generated ImplementationProperty."""
        for trname, attrname in self.transitions_at.items():

            implem = self.implementations[trname]

            if attrname in attrs:
                conflicting = attrs[attrname]
                if not self._may_override(implem, conflicting):
                    raise ValueError(
                        "Can't override transition implementation %s=%r with %r" %
                        (attrname, conflicting, implem))

            attrs[attrname] = implem
        return attrs