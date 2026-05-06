def parse_400_row(row: list) -> tuple:
    """ Interval event record (400) """

    return EventRecord(int(row[1]), int(row[2]), row[3], row[4], row[5])