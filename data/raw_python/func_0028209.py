def get_identifiers_from_input_file(input_file_name):
    """Read identifiers from input file into list of dicts with the header row values
       as keys, and the rest of the rows as values.
    """
    valid_identifiers = ['address', 'zipcode', 'unit', 'city', 'state', 'slug', 'block_id', 'msa',
                         'num_bins', 'property_type', 'client_value', 'client_value_sqft', 'meta']
    mode = 'r'
    if sys.version_info[0] < 3:
        mode = 'rb'
    with io.open(input_file_name, mode) as input_file:
        result = [{identifier: val for identifier, val in list(row.items())
                   if identifier in valid_identifiers}
                  for row in csv.DictReader(input_file, skipinitialspace=True)]
        return result