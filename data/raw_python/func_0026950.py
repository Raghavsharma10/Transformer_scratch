def print_meter_record(file_path, rows=5):
    """ Output readings for specified number of rows to console """
    m = nr.read_nem_file(file_path)
    print('Header:', m.header)
    print('Transactions:', m.transactions)
    for nmi in m.readings:
        for channel in m.readings[nmi]:
            print(nmi, 'Channel', channel)
            for reading in m.readings[nmi][channel][-rows:]:
                print('', reading)