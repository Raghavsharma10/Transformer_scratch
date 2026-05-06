def index_from_id(self,Id):
        """Return the row of given Id if it'exists, otherwise None. Only works with pseudo-acces"""
        try:
            return [a.Id for a in self].index(Id)
        except IndexError:
            return