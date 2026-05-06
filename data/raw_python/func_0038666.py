def quick_search(self, terms):
        '''Wrapper for search_bugs, for simple string searches'''
        assert type(terms) is str
        p = [{'quicksearch': terms}]
        return self.search_bugs(p)