def add(db, anime):
    """Add an anime (or update existing).

    anime is an AnimeTree instance.

    """
    aid = anime.aid
    values = {
        'aid': aid,
        'title': anime.title,
        'type': anime.type,
        'episodecount': anime.episodecount,
    }
    if anime.startdate is not None:
        values['startdate'] = datets.to_ts(anime.startdate)
    if anime.enddate is not None:
        values['enddate'] = datets.to_ts(anime.enddate)
    with db:
        upsert(db, 'anime', ['aid'], values)
        our_anime = lookup(db, anime.aid, episode_fields=ALL)
        our_episodes = our_anime.episodes
        for episode in anime.episodes:
            add_episode(db, aid, episode)
            our_episodes = [
                ep for ep in our_episodes
                if not (ep.type == episode.type and ep.number == episode.number)
            ]
        # Remove extra episodes that we have.
        for episode in our_episodes:
            delete_episode(db, aid, episode)