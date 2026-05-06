def get_tier_from_participants(participantsIdentities, minimum_tier=Tier.bronze, queue=Queue.RANKED_SOLO_5x5):
    """
    Returns the tier of the lowest tier and the participantsIDs divided by tier
    player in the match
    :param participantsIdentities: the match participants
    :param minimum_tier: the minimum tier that a participant must be in order to be added
    :param queue: the queue over which the tier of the player is considered
    :return: the tier of the lowest tier player in the match
    """
    leagues = leagues_by_summoner_ids([p.player.summonerId for p in participantsIdentities], queue)
    match_tier = max(leagues.keys(), key=operator.attrgetter('value'))
    return match_tier, {league: ids for league, ids in leagues.items() if league.is_better_or_equal(minimum_tier)}