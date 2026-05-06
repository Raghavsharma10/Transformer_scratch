def allready(self, obj) -> bool:
        """Return |True| or |False| to indicate whether all protected
        properties are ready or not."""
        for prop in self.__properties:
            if not prop.isready(obj):
                return False
        return True