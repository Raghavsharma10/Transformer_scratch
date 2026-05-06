async def get_final_ranking(self) -> OrderedDict:
        """ Get the ordered players ranking

        Returns:
            collections.OrderedDict[rank, List[Participant]]:

        Raises:
            APIException

        """
        if self._state != TournamentState.complete.value:
            return None

        ranking = {}
        for p in self.participants:
            if p.final_rank in ranking:
                ranking[p.final_rank].append(p)
            else:
                ranking[p.final_rank] = [p]

        return OrderedDict(sorted(ranking.items(), key=lambda t: t[0]))