def database_to_excel(engine, excel_file_path):
    """Export database to excel.

    :param engine: 
    :param excel_file_path:
    """
    from sqlalchemy import MetaData, select

    metadata = MetaData()
    metadata.reflect(engine)

    writer = pd.ExcelWriter(excel_file_path)
    for table in metadata.tables.values():
        sql = select([table])
        df = pd.read_sql(sql, engine)
        df.to_excel(writer, table.name, index=False)

    writer.save()