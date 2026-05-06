def get_parsed_cells(iw_data, rules=None):
    """ Parses iwlist output into a list of networks.
        @param list iw_data
            Output from iwlist scan.
            A list of strings.

        @return list
            properties: Name, Address, Quality, Channel, Frequency, Encryption, Signal Level, Noise Level, Bit Rates, Mode.
    """

    # Here's a dictionary of rules that will be applied to the description
    # of each cell. The key will be the name of the column in the table.
    # The value is a function defined above.
    rules = rules or {
        "Name": get_name,
        "Quality": get_quality,
        "Channel": get_channel,
        "Frequency": get_frequency,
        "Encryption": get_encryption,
        "Address": get_address,
        "Signal Level": get_signal_level,
        "Noise Level": get_noise_level,
        "Bit Rates": get_bit_rates,
        "Mode": get_mode,
    }

    cells = [[]]
    parsed_cells = []

    for line in iw_data:
        cell_line = match(line, "Cell ")
        if cell_line != None:
            cells.append([])
            line = cell_line[-27:]
        cells[-1].append(line.rstrip())

    cells = cells[1:]

    for cell in cells:
        parsed_cells.append(parse_cell(cell, rules))

    sort_cells(parsed_cells)
    return parsed_cells