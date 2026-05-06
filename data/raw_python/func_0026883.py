def parse_200_row(row: list) -> NmiDetails:
    """ Parse NMI data details record (200) """
    return NmiDetails(row[1], row[2], row[3], row[4], row[5], row[6],
                         row[7], int(row[8]), parse_datetime(row[9]))