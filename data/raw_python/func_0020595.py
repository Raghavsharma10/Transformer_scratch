def matchup(self):
        """
        Return the game meta information displayed in report banners including team names,
        final score, game date, location, and attendance. Data format is
        
        .. code:: python
        
            {
                'home': home,
                'away': away,
                'final': final,
                'attendance': att,
                'date': date,
                'location': loc
            }
            
        :returns: matchup banner info
        :rtype: dict
        """
        if self.play_by_play.matchup:
            return self.play_by_play.matchup
        elif self.rosters.matchup:
            return self.rosters.matchup
        elif self.toi.matchup:
            return self.toi.matchup
        else:
            self.face_off_comp.matchup