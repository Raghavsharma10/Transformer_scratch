def validate_header(header, required_fields=None):
    '''validate_header ensures that the first row contains the exp_id,
       var_name, var_value, and token. Capitalization isn't important, but
       ordering is. This criteria is very strict, but it's reasonable
       to require.
 
       Parameters
       ==========
       header: the header row, as a list
       required_fields: a list of required fields. We derive the required
                        length from this list.

       Does not return, instead exits if malformed. Runs silently if OK.

    '''
    if required_fields is None:
        required_fields = ['exp_id', 'var_name', 'var_value', 'token']

    # The required length of the header based on required fields

    length = len(required_fields)

    # This is very strict, but no reason not to be

    header = _validate_row(header, required_length=length) 
    header = [x.lower() for x in header]

    for idx in range(length):
        field = header[idx].lower().strip()
        if required_fields[idx] != field:
            bot.error('Malformed header field %s, exiting.' %field)
            sys.exit(1)