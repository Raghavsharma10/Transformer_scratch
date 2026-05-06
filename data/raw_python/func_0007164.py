def get_info(self, key=None, Id=None) -> dict:
        """Returns information associated with Id or list index"""
        if key is not None:
            Id = self[key].Id
        return self.infos.get(Id,{})