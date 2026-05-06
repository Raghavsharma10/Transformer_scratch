def parse(db_folder, out_folder):
    """
    Parse a cronos database.

    Convert the database located in ``db_folder`` into CSV files in the
    directory ``out_folder``.
    """
    # The database structure, containing table and column definitions as
    # well as other data.
    stru_dat = get_file(db_folder, 'CroStru.dat')
    # Index file for the database, which contains offsets for each record.
    data_tad = get_file(db_folder, 'CroBank.tad')
    # Actual data records, can only be decoded using CroBank.tad.
    data_dat = get_file(db_folder, 'CroBank.dat')
    if None in [stru_dat, data_tad, data_dat]:
        raise CronosException("Not all database files are present.")

    meta, tables = parse_structure(stru_dat)

    for table in tables:
        # TODO: do we want to export the "FL" table?
        if table['abbr'] == 'FL' and table['name'] == 'Files':
            continue
        fh = open(make_csv_file_name(meta, table, out_folder), 'w')
        columns = table.get('columns')
        writer = csv.writer(fh)
        writer.writerow([encode_cell(c['name']) for c in columns])
        for row in parse_data(data_tad, data_dat, table.get('id'), columns):
            writer.writerow([encode_cell(c) for c in row])
        fh.close()