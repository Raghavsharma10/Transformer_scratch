def get_epno(self, episode: Episode):
        """Return epno for an Episode instance.

        epno is a string formatted with the episode number and type, e.g., S1,
        T2.

        >>> x = EpisodeTypes([EpisodeType(1, 'foo', 'F')])
        >>> ep = Episode(type=1, number=2)
        >>> x.get_epno(ep)
        'F2'

        """
        return '{}{}'.format(self[episode.type].prefix, episode.number)