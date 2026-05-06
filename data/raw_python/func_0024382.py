def reset_query_marks(self):
        """
        set or reset hyb and neighbors marks to atoms.
        """
        for ml in (self.__reagents, self.__reactants, self.__products):
            for m in ml:
                if hasattr(m, 'reset_query_marks'):
                    m.reset_query_marks()
        self.flush_cache()