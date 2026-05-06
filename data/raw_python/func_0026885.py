def parse_300_row(row: list, interval: int, uom: str) -> IntervalRecord:
    """ Interval data record (300) """

    num_intervals = int(24 * 60 / interval)
    interval_date = parse_datetime(row[1])
    last_interval = 2 + num_intervals
    quality_method = row[last_interval]

    interval_values = parse_interval_records(
        row[2:last_interval], interval_date, interval, uom, quality_method)

    return IntervalRecord(interval_date, interval_values,
                             row[last_interval + 0], row[last_interval + 1],
                             row[last_interval + 2],
                             parse_datetime(row[last_interval + 3]),
                             parse_datetime(row[last_interval + 4]))