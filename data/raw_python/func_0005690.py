def validate(self, obj):
        """check if obj has this api param"""
        if self.path:
            for i in self.path:
                obj = obj[i]
        obj = obj[self.field]

        raise NotImplementedError('Validation is not implemented yet')