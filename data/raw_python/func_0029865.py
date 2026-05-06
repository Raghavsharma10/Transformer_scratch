def update(self, f):
        """Copy another files properties into this one."""

        for p in self.__mapper__.attrs:

            if p.key == 'oid':
                continue
            try:
                setattr(self, p.key, getattr(f, p.key))

            except AttributeError:
                # The dict() method copies data property values into the main dict,
                # and these don't have associated class properties.
                continue