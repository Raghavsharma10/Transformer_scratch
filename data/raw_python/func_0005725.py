def get_fault(fault_id=None):
    """Retrieve a randomly-generated error message as a unicode string.
    
    :param fault_id:
        
        Allows you to optionally specify an integer representing the fault_id 
        from the database table.  This allows you to retrieve a specific fault
        each time, albeit with different keywords."""

    counts = __get_table_limits()
    result = None
    id_ = 0

    try:
        if isinstance(fault_id, int):
            id_ = fault_id
        elif isinstance(fault_id, float):
            print("""ValueError:  Floating point number detected.
                  Rounding number to 0 decimal places.""")
            id_ = round(fault_id)
        else:
            id_ = random.randint(1, counts['max_fau'])

    except ValueError:
        print("ValueError:  Incorrect parameter type detected.")

    if id_ <= counts['max_fau']:
        fault = __get_fault(counts, fault_id=id_)
    else:
        print("""ValueError:  Parameter integer is too high.
              Maximum permitted value is {0}.""".format(str(counts['max_fau'])))
        id_ = counts['max_fau']
        fault = __get_fault(counts, fault_id=id_)

    if fault is not None:
        while fault[0] == 'n':
            if id_ is not None:
                fault = __get_fault(counts, None)
            else:
                fault = __get_fault(counts, id_)
        if fault[0] == 'y':
            result = __process_sentence(fault, counts)
        return result
    else:
        print('ValueError: _fault cannot be None.')