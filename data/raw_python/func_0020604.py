def parse(self):
        """
        Retreive and parse Play by Play data for the given :py:class:`nhlscrapi.games.game.GameKey``

        :returns: ``self`` on success, ``None`` otherwise
        """

        try:
            return super(RosterRep, self).parse() \
                .parse_rosters() \
                .parse_scratches() \
                .parse_coaches() \
                .parse_officials()
        except:
            return None