def insert_paraphrase_information(germanet_db, wiktionary_files):
    '''
    Reads in the given GermaNet relation file and inserts its contents
    into the given MongoDB database.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    - `wiktionary_files`:
    '''
    num_paraphrases = 0
    # cache the lexunits while we work on them
    lexunits = {}
    for filename in wiktionary_files:
        paraphrases = read_paraphrase_file(filename)
        num_paraphrases += len(paraphrases)
        for paraphrase in paraphrases:
            if paraphrase['lexUnitId'] not in lexunits:
                lexunits[paraphrase['lexUnitId']] = \
                    germanet_db.lexunits.find_one(
                    {'id': paraphrase['lexUnitId']})
            lexunit = lexunits[paraphrase['lexUnitId']]
            if 'paraphrases' not in lexunit:
                lexunit['paraphrases'] = []
            lexunit['paraphrases'].append(paraphrase)
    for lexunit in lexunits.values():
        germanet_db.lexunits.save(lexunit)

    print('Inserted {0} wiktionary paraphrases.'.format(num_paraphrases))