def __get_table_limits():
    """Here we simply take a count of each of the database tables so we know our
    upper limits for our random number calls then return a dictionary of them 
    to the calling function..."""

    table_counts = {
        'max_adjectives': None,
        'max_names': None,
        'max_nouns': None,
        'max_sentences': None,
        'max_faults': None,
        'max_verbs': None
    }

    cursor = CONN.cursor()

    cursor.execute('SELECT count(*) FROM suradjs')
    table_counts['max_adjectives'] = cursor.fetchone()
    table_counts['max_adjectives'] = table_counts['max_adjectives'][0]

    cursor.execute('SELECT count(*) FROM surnames')
    table_counts['max_names'] = cursor.fetchone()
    table_counts['max_names'] = table_counts['max_names'][0]

    cursor.execute('SELECT count(*) FROM surnouns')
    table_counts['max_nouns'] = cursor.fetchone()
    table_counts['max_nouns'] = table_counts['max_nouns'][0]

    cursor.execute('SELECT count(*) FROM sursentences')
    table_counts['max_sen'] = cursor.fetchone()
    table_counts['max_sen'] = table_counts['max_sen'][0]

    cursor.execute('SELECT count(*) FROM surfaults')
    table_counts['max_fau'] = cursor.fetchone()
    table_counts['max_fau'] = table_counts['max_fau'][0]

    cursor.execute('SELECT count(*) FROM surverbs')
    table_counts['max_verb'] = cursor.fetchone()
    table_counts['max_verb'] = table_counts['max_verb'][0]

    return table_counts