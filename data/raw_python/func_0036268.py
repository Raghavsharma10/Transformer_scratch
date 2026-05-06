def select(
        db,
        where_query: str,
        where_params: SQLParams,
        fields: FieldsParam = ALL,
        episode_fields: FieldsParam = (),
) -> Iterator[Anime]:
    """Perform an arbitrary SQL SELECT WHERE on the anime table.

    By nature of "arbitrary query", this is vulnerable to injection, use only
    trusted values for `where_query`.

    This will "lazily" fetch the requested fields as needed.  For example,
    episodes (which require a separate query per anime) will only be fetched if
    `episode_fields` is provided.  Anime status will be cached only if status
    fields are requested.

    :param str where_query: SELECT WHERE query
    :param where_params: parameters for WHERE query
    :param fields: anime fields to get.  If :const:`ALL`, get all fields.
        Default is :const:`ALL`.
    :param episode_fields: episode fields to get.
        If :const:`ALL`, get all fields.  If empty, don't get episodes.
        `fields` must contain 'aid' to get episodes.
    :param bool force_status: whether to force status calculation.
    :returns: iterator of Anime

    """

    logger.debug(
        'select(%r, %r, %r, %r, %r)',
        db, where_query, where_params, fields, episode_fields)
    fields = _clean_fields(ANIME_FIELDS, fields)
    if not fields:
        raise ValueError('Fields cannot be empty')
    if set(fields) & STATUS_FIELDS.keys():
        cur = db.cursor().execute(
            ANIME_QUERY.format('aid', where_query),
            where_params)
        for row in cur:
            cache_status(db, row[0])

    if 'aid' in fields:
        episode_fields = _clean_fields(EPISODE_FIELDS, episode_fields)
    else:
        episode_fields = ()

    with db:
        anime_query = ANIME_QUERY.format(
            ','.join(ANIME_FIELDS[field] for field in fields),
            where_query,
        )
        anime_rows = db.cursor().execute(anime_query, where_params)
        for row in anime_rows:
            anime = Anime(**{
                field: value
                for field, value in zip(fields, row)})

            if episode_fields:
                episode_query = 'SELECT {} FROM episode WHERE aid=?'
                episode_query = episode_query.format(
                    ','.join(EPISODE_FIELDS[field] for field in episode_fields))

                episode_rows = db.cursor().execute(episode_query, (anime.aid,))
                episodes = [
                    Episode(**{
                        field: value
                        for field, value in zip(episode_fields, row)})
                    for row in episode_rows]
                anime.episodes = episodes

            yield anime