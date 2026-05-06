def get_sentence(sentence_id=None):
    """Retrieve a randomly-generated sentence as a unicode string.
    
    :param sentence_id:
        
        Allows you to optionally specify an integer representing the sentence_id
        from the database table.  This allows you to retrieve a specific
        sentence each time, albeit with different keywords."""

    counts = __get_table_limits()
    result = None
    id_ = 0

    try:
        if isinstance(sentence_id, int):
            id_ = sentence_id
        elif isinstance(sentence_id, float):
            print("""ValueError:  Floating point number detected.
                  Rounding number to 0 decimal places.""")
            id_ = round(sentence_id)
        else:
            id_ = random.randint(1, counts['max_sen'])

    except ValueError:
        print("ValueError:  Incorrect parameter type detected.")

    if id_ <= counts['max_sen']:
        sentence = __get_sentence(counts, sentence_id=id_)
    else:
        print("""ValueError:  Parameter integer is too high.
              Maximum permitted value is {0}.""".format(str(counts['max_sen'])))
        id_ = counts['max_sen']
        sentence = __get_sentence(counts, sentence_id=id_)

    if sentence is not None:
        while sentence[0] == 'n':
            if id_ is not None:
                # here we delibrately pass 'None' to __getsentence__ as it will
                sentence = __get_sentence(counts, None)
            else:
                sentence = __get_sentence(counts, id_)
        if sentence[0] == 'y':
            result = __process_sentence(sentence, counts)
        return result
    else:
        print('ValueError: _sentence cannot be None.')