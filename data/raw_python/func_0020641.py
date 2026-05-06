def parse(self):
        """
        Retreive and parse Event Summary report for the given :py:class:`nhlscrapi.games.game.GameKey`
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            return super(EventSummRep, self).parse() \
                .parse_away_shots() \
                .parse_home_shots() \
                .parse_away_fo() \
                .parse_home_fo() \
                .parse_away_by_player() \
                .parse_home_by_player()
        except:
            return None