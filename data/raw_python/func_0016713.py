def insert_lemmatisation_data(germanet_db):
    '''
    Creates the lemmatiser collection in the given MongoDB instance
    using the data derived from the Projekt deutscher Wortschatz.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    '''
    # drop the database collection if it already exists
    germanet_db.lemmatiser.drop()
    num_lemmas = 0
    input_file = gzip.open(os.path.join(os.path.dirname(__file__),
                                        LEMMATISATION_FILE))
    for line in input_file:
        line = line.decode('iso-8859-1').strip().split('\t')
        assert len(line) == 2
        germanet_db.lemmatiser.insert(dict(list(zip(('word', 'lemma'), line))))
        num_lemmas += 1
    input_file.close()
    # index the collection on 'word'
    germanet_db.lemmatiser.create_index('word')

    print('Inserted {0} lemmatiser entries.'.format(num_lemmas))