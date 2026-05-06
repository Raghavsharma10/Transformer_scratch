def set_search_engine(self, engine):
        '''Set the search ``engine`` for this :class:`Router`.'''
        self._search_engine = engine
        self._search_engine.set_router(self)