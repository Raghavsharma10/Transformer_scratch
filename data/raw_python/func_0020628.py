def update(self, play):
        """
        Update the accumulator with the current play
        
        :returns: new tally
        :rtype: dict, ``{ 'period': per, 'time': clock, 'team': cumul, 'play': play }``
        """
        new_tally = { }
        #if any(isinstance(play.event, te) for te in self.trigger_event_types):
        if self._count_play(play):
            # the team who made the play / triggered the event
    
            team = self._get_team(play)

            try:
                self.total[team] += 1
            except:
                self.total[team] = 1
                self.teams.append(team)
                for i in range(len(self.tally)):
                    self.tally[i][team] = 0
      
            try:
                new_tally = { k:v for k,v in self.tally[len(self.tally)-1].items() }
                new_tally['period'] = play.period
                new_tally['time'] = play.time
                new_tally[team] += 1
                new_tally['play'] = play
            except:
                new_tally = {
                    'period': play.period,
                    'time': play.time,
                    team: 1,
                    'play': play
                }
      
            self.tally.append(new_tally)
      
        return new_tally