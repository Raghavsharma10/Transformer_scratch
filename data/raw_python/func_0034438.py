def asDict(self):
        """Returns a serializable object"""
        return {
            'isError': self.isError,
            'message': self.message,
            'values': self.values,
            'value': self.value,
        }