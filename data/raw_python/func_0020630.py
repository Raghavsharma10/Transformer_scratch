def compute_stats(self):
        """
        Compute the stats defined in ``self.cum_stats``.
        
        :returns: collection of all computed :py:class:`.AccumulateStats`
        :rtype: dict
        """
        if not self.__have_stats:
            if self.init_cs_teams and self.cum_stats:
                self.__init_cs_teams()
            
            for play in self._rep_reader.parse_plays_stream():
                p = Play(**play)
                self.__wrapped_plays.append(p)
                if self.cum_stats:
                    self.__process(p, self.cum_stats, 'update')
                self.__have_stats = True
                
        return self.cum_stats