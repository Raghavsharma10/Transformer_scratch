def parse_100_row(row: list, file_name: str) -> HeaderRecord:
    """ Parse header record (100) """
    return HeaderRecord(
        row[1],
        parse_datetime(row[2]),
        row[3],
        row[4],
        file_name,
    )