def extend(self, collection):
        """Merges collections. Ensure uniqueness of ids"""
        l_ids = set([a.Id for a in self])
        for acces in collection:
            if not acces.Id in l_ids:
                list.append(self,acces)
                info = collection.get_info(Id=acces.Id)
                if info:
                    self.infos[acces.Id] = info