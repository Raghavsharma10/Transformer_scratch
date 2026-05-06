def db_export(schema, uuid, object_filter, export_format, filename, pretty, all_schemata, omit):
    """Export stored objects

    Warning! This functionality is work in progress and you may destroy live data by using it!
    Be very careful when using the export/import functionality!"""

    internal_backup(schema, uuid, object_filter, export_format, filename, pretty, all_schemata, omit)