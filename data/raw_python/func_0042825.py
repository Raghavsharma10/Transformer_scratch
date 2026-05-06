def search_files_sql_builder(search):
    """
    Create and populate an instance of :class:`meteorpi_db.SQLBuilder` for a given
    :class:`meteorpi_model.FileRecordSearch`. This can then be used to retrieve the results of the search, materialise
    them into :class:`meteorpi_model.FileRecord` instances etc.

    :param FileRecordSearch search:
        The search to realise
    :return:
        A :class:`meteorpi_db.SQLBuilder` configured from the supplied search
    """
    b = SQLBuilder(tables="""archive_files f
INNER JOIN archive_semanticTypes s2 ON f.semanticType=s2.uid
INNER JOIN archive_observations o ON f.observationId=o.uid
INNER JOIN archive_semanticTypes s ON o.obsType=s.uid
INNER JOIN archive_observatories l ON o.observatory=l.uid""", where_clauses=[])
    b.add_set_membership(search.obstory_ids, 'l.publicId')
    b.add_sql(search.repository_fname, 'f.repositoryFname = %s')
    b.add_sql(search.observation_type, 's.name = %s')
    b.add_sql(search.observation_id, 'o.uid = %s')
    b.add_sql(search.time_min, 'f.fileTime > %s')
    b.add_sql(search.time_max, 'f.fileTime < %s')
    b.add_sql(search.lat_min, 'l.latitude >= %s')
    b.add_sql(search.lat_max, 'l.latitude <= %s')
    b.add_sql(search.long_min, 'l.longitude >= %s')
    b.add_sql(search.long_max, 'l.longitude <= %s')
    b.add_sql(search.mime_type, 'f.mimeType = %s')
    b.add_sql(search.semantic_type, 's2.name = %s')
    b.add_metadata_query_properties(meta_constraints=search.meta_constraints, id_column="fileId", id_table="f")

    # Check for import / export filters
    if search.exclude_imported:
        b.where_clauses.append('NOT EXISTS (SELECT * FROM archive_observationImport i WHERE i.observationId = o.uid')
    if search.exclude_export_to is not None:
        b.where_clauses.append("""
        NOT EXISTS (SELECT * FROM archive_fileExport ex
        INNER JOIN archive_exportConfig c ON ex.exportConfig = c.uid
        WHERE ex.fileId = f.uid  AND c.exportConfigID = %s)
        """)
        b.sql_args.append(SQLBuilder.map_value(search.exclude_export_to))

    return b