def fetch_csv_db(
        table_name,
        download_url,
        csv_filename=None,
        db_filename=None,
        subdir=None,
        version=1,
        **pandas_kwargs):
    """
    Download a remote CSV file and create a local sqlite3 database
    from its contents
    """
    df = fetch_csv_dataframe(
        download_url=download_url,
        filename=csv_filename,
        subdir=subdir,
        **pandas_kwargs)
    base_filename = splitext(csv_filename)[0]
    if db_filename is None:
        db_filename = _db_filename_from_dataframe(base_filename, df)
    return db_from_dataframe(
        db_filename,
        table_name,
        df,
        subdir=subdir,
        version=version)