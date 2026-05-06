def participants(self):
        """agents + computers (i.e. all non-observers)"""
        ret = []
        for p in self.players:
            try:
                if     p.isComputer: ret.append(p)
                if not p.isObserver: ret.append(p) # could cause an exception if player isn't a PlayerPreGame
            except AttributeError: pass
        return ret