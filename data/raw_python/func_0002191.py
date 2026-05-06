def parse_csv_response(data, unit_handler):
    """Handle CSV-formatted HTTP responses."""
    return squish([parse_csv_dataset(d, unit_handler) for d in data.split(b'\n\n')])