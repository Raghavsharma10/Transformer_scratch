def _get_extra(self, attrs, exclude):
        """Read the extra properties, taking into account an exclude list"""
        result = {}
        for key in attrs.getNames():
            if key not in exclude:
                result[str(key)] = str(attrs[key])
        return result