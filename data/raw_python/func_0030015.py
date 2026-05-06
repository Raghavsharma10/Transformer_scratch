def is_label(self):
        """Return true if the colum is a dimension"""
        from ambry.valuetype.core import ROLE
        return self.role == ROLE.LABEL