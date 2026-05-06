def remove_id(self,key):
        """Suppress acces with id = key"""
        self.infos.pop(key, "")
        new_l = [a for a in self if not (a.Id == key)]
        list.__init__(self, new_l)