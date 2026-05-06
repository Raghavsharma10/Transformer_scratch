def search_metadata_sql_builder(search):
    """
    Create and populate an instance of :class:`meteorpi_db.SQLBuilder` for a given
    :class:`meteorpi_model.ObservatoryMetadataSearch`. This can then be used to retrieve the results of the search,
    materialise them into :class:`meteorpi_model.ObservatoryMetadata` instances etc.

    :param ObservatoryMetadataSearch search:
        The search to realise
    :return:
        A :class:`meteorpi_db.SQLBuilder` configured from the supplied search
    """
    b = SQLBuilder(tables="""archive_metadata m
INNER JOIN archive_metadataFields f ON m.fieldId=f.uid
INNER JOIN archive_observatories l ON m.observatory=l.uid""", where_clauses=["m.observatory IS NOT NULL"])
    b.add_set_membership(search.obstory_ids, 'l.publicId')
    b.add_sql(search.field_name, 'f.metaKey = %s')
    b.add_sql(search.time_min, 'm.time > %s')
    b.add_sql(search.time_max, 'm.time < %s')
    b.add_sql(search.lat_min, 'l.latitude >= %s')
    b.add_sql(search.lat_max, 'l.latitude <= %s')
    b.add_sql(search.long_min, 'l.longitude >= %s')
    b.add_sql(search.long_max, 'l.longitude <= %s')
    b.add_sql(search.item_id, 'm.publicId = %s')

    # Check for import / export filters
    if search.exclude_imported:
        b.where_clauses.append('NOT EXISTS (SELECT * FROM archive_metadataImport i WHERE i.metadataId = m.uid')
    if search.exclude_export_to is not None:
        b.where_clauses.append("""
        NOT EXISTS (SELECT * FROM archive_metadataExport ex
        INNER JOIN archive_exportConfig c ON ex.exportConfig = c.uid
        WHERE ex.metadataId = m.uid AND c.exportConfigID = %s)
        """)
        b.sql_args.append(SQLBuilder.map_value(search.exclude_export_to))

    return b