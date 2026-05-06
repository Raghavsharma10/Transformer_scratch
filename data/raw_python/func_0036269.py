def lookup(
        db,
        aid: int,
        fields: FieldsParam = ALL,
        episode_fields: FieldsParam = (),
) -> Anime:
    """Look up information for a single anime.

    :param fields: anime fields to get.  If ``None``, get all fields.
    :param episode_fields: episode fields to get.
        If ``None``, get all fields.  If empty, don't get episodes.

    """
    return next(select(
        db,
        'aid=?',
        (aid,),
        fields=fields,
        episode_fields=episode_fields,
    ))