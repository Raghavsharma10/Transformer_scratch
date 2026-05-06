def parse_250_row(row: list) -> BasicMeterData:
    """ Parse basic meter data record (250) """
    return BasicMeterData(row[1], row[2], row[3], row[4], row[5],
                             row[6], row[7], float(row[8]),
                             parse_datetime(row[9]), row[10], row[11], row[12],
                             float(row[13]), parse_datetime(
                                 row[14]), row[15], row[16], row[17],
                             float(row[18]), row[19], row[20],
                             parse_datetime(row[21]), parse_datetime(row[22]))