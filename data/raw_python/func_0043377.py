def postadressen(self):
        '''
        Returns the postadressen for this Perceel.

        Will only take the huisnummers with status `inGebruik` into account.

        :rtype: list 
        '''
        return [h.postadres for h in self.huisnummers if h.status.id == '3']