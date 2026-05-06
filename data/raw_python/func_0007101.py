def modifie_many(self, dic: dict):
        """Convenience function which calls modifie on each element of dic"""
        for i, v in dic.items():
            self.modifie(i, v)