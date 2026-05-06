def get_addresses_from_input_file(input_file_name):
    """Read addresses from input file into list of tuples.
       This only supports address and zipcode headers
    """
    mode = 'r'
    if sys.version_info[0] < 3:
        mode = 'rb'
    with io.open(input_file_name, mode) as input_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')

        addresses = list(map(tuple, reader))

        if len(addresses) == 0:
            raise Exception('No addresses found in input file')

        header_columns = list(column.lower() for column in addresses.pop(0))

        try:
            address_index = header_columns.index('address')
            zipcode_index = header_columns.index('zipcode')
        except ValueError:
            raise Exception("""The first row of the input CSV must be a header that contains \
a column labeled 'address' and a column labeled 'zipcode'.""")

        return list((row[address_index], row[zipcode_index]) for row in addresses)