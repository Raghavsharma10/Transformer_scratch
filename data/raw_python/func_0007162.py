def append(self, acces, **kwargs):
        """Append acces to list. Quite slow since it checks uniqueness.
        kwargs may set `info` for this acces.
        """
        if acces.Id in set(ac.Id for ac in self):
            raise ValueError("Acces id already in list !")
        list.append(self, acces)
        if kwargs:
            self.infos[acces.Id] = kwargs