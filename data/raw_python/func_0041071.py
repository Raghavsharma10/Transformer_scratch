def from_dict(cls, data):
        """Converts this from a dictionary to a object."""
        data = dict(data)
        cause = data.get('cause')
        if cause is not None:
            data['cause'] = cls.from_dict(cause)
        return cls(**data)