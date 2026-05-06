def _production(self):
        """Calculate total energy production. Not rounded"""
        return self._nuclear + self._diesel + self._gas + self._wind + self._combined + self._vapor + self._solar + self._hydraulic + self._carbon + self._waste +  self._other