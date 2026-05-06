def leagues_by_summoner_ids(summoner_ids, queue=Queue.RANKED_SOLO_5x5):
    """
    Takes in a list of players ids and divide them by league tiers.
    :param summoner_ids: a list containing the ids of players
    :param queue: the queue to consider
    :return: a dictionary tier -> set of ids
    """
    summoners_league = defaultdict(set)
    for start, end in _slice(0, len(summoner_ids), 10):
        for id, leagues in get_league_entries_by_summoner(summoner_ids[start:end]).items():
            for league in leagues:
                if Queue[league.queue]==queue:
                    summoners_league[Tier.parse(league.tier)].add(int(id))
    return summoners_league