def insert_lexical_information(germanet_db, lex_files):
    '''
    Reads in the given lexical information files and inserts their
    contents into the given MongoDB database.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    - `lex_files`: a list of paths to XML files containing lexial
      information
    '''
    # drop the database collections if they already exist
    germanet_db.lexunits.drop()
    germanet_db.synsets.drop()
    # inject data from XML files into the database
    for lex_file in lex_files:
        synsets = read_lexical_file(lex_file)
        for synset in synsets:
            synset = dict((SYNSET_KEY_REWRITES.get(key, key), value)
                          for (key, value) in synset.items())
            lexunits = synset['lexunits']
            synset['lexunits'] = germanet_db.lexunits.insert(lexunits)
            synset_id = germanet_db.synsets.insert(synset)
            for lexunit in lexunits:
                lexunit['synset']   = synset_id
                lexunit['category'] = synset['category']
                germanet_db.lexunits.save(lexunit)
    # index the two collections by id
    germanet_db.synsets.create_index('id')
    germanet_db.lexunits.create_index('id')
    # also index lexunits by lemma, lemma-pos, and lemma-pos-sensenum
    germanet_db.lexunits.create_index([('orthForm', DESCENDING)])
    germanet_db.lexunits.create_index([('orthForm', DESCENDING),
                                       ('category', DESCENDING)])
    germanet_db.lexunits.create_index([('orthForm', DESCENDING),
                                       ('category', DESCENDING),
                                       ('sense', DESCENDING)])
    print('Inserted {0} synsets, {1} lexical units.'.format(
        germanet_db.synsets.count(),
        germanet_db.lexunits.count()))