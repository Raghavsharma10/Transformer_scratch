def get_kind(self, value):
        """Return the kind (type) of the attribute"""
        if isinstance(value, float):
            return 'f'
        elif isinstance(value, int):
            return 'i'
        else:
            raise ValueError("Only integer or floating point values can be stored.")