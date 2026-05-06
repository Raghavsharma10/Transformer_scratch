def get(self, attr, default=None):
        """Get an attribute defined by this session"""

        attrs = self.body.get('attributes') or {}
        return attrs.get(attr, default)