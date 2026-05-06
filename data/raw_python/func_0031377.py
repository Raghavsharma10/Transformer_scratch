def get_one(self, criteria):
        ''' return one item
        '''
        try:
            items = [item for item in self._get_with_criteria(criteria, limit=1)]
            return items[0]
        except:
            return None