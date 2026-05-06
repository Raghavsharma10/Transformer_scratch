def search_obsgroups_sql_builder(search):
    """
    Create and populate an instance of :class:`meteorpi_db.SQLBuilder` for a given
    :class:`meteorpi_model.ObservationGroupSearch`. This can then be used to retrieve the results of the search,
    materialise them into :class:`meteorpi_model.ObservationGroup` instances etc.

    :param ObservationGroupSearch search:
        The search to realise
    :return:
        A :class:`meteorpi_db.SQLBuilder` configured from the supplied search
    """
    b = SQLBuilder(tables="""archive_obs_groups g
INNER JOIN archive_semanticTypes s ON g.semanticType=s.uid""", where_clauses=[])
    b.add_sql(search.obstory_name, """
EXISTS (SELECT 1 FROM archive_obs_group_members x1
INNER JOIN archive_observations x2 ON x2.uid=x1.observationId
INNER JOIN archive_observatories x3 ON x3.uid=x2.observatory
WHERE x1.groupId=g.uid AND x3.publicId=%s)""")
    b.add_sql(search.semantic_type, 's.name = %s')
    b.add_sql(search.observation_id, """
EXISTS (SELECT 1 FROM archive_obs_group_members y1
INNER JOIN archive_observations y2 ON y2.uid=y1.observationId
WHERE y1.groupId=g.uid AND y2.publicId=%s)""")
    b.add_sql(search.group_id, 'g.publicId = %s')
    b.add_sql(search.time_min, 'g.time > %s')
    b.add_sql(search.time_max, 'g.time < %s')
    b.add_metadata_query_properties(meta_constraints=search.meta_constraints, id_column="groupId", id_table="g")
    return b