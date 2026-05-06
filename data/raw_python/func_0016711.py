def insert_relation_information(germanet_db, gn_rels_file):
    '''
    Reads in the given GermaNet relation file and inserts its contents
    into the given MongoDB database.

    Arguments:
    - `germanet_db`: a pymongo.database.Database object
    - `gn_rels_file`:
    '''
    lex_rels, con_rels = read_relation_file(gn_rels_file)

    # cache the lexunits while we work on them
    lexunits = {}
    for lex_rel in lex_rels:
        if lex_rel['from'] not in lexunits:
            lexunits[lex_rel['from']] = germanet_db.lexunits.find_one(
                {'id': lex_rel['from']})
        from_lexunit = lexunits[lex_rel['from']]
        if lex_rel['to'] not in lexunits:
            lexunits[lex_rel['to']] = germanet_db.lexunits.find_one(
                {'id': lex_rel['to']})
        to_lexunit = lexunits[lex_rel['to']]
        if 'rels' not in from_lexunit:
            from_lexunit['rels'] = set()
        from_lexunit['rels'].add((lex_rel['name'], to_lexunit['_id']))
        if lex_rel['dir'] == 'both':
            if 'rels' not in to_lexunit:
                to_lexunit['rels'] = set()
            to_lexunit['rels'].add((lex_rel['inv'], from_lexunit['_id']))
    for lexunit in lexunits.values():
        if 'rels' in lexunit:
            lexunit['rels'] = sorted(lexunit['rels'])
            germanet_db.lexunits.save(lexunit)

    # cache the synsets while we work on them
    synsets = {}
    for con_rel in con_rels:
        if con_rel['from'] not in synsets:
            synsets[con_rel['from']] = germanet_db.synsets.find_one(
                {'id': con_rel['from']})
        from_synset = synsets[con_rel['from']]
        if con_rel['to'] not in synsets:
            synsets[con_rel['to']] = germanet_db.synsets.find_one(
                {'id': con_rel['to']})
        to_synset = synsets[con_rel['to']]
        if 'rels' not in from_synset:
            from_synset['rels'] = set()
        from_synset['rels'].add((con_rel['name'], to_synset['_id']))
        if con_rel['dir'] in ['both', 'revert']:
            if 'rels' not in to_synset:
                to_synset['rels'] = set()
            to_synset['rels'].add((con_rel['inv'], from_synset['_id']))
    for synset in synsets.values():
        if 'rels' in synset:
            synset['rels'] = sorted(synset['rels'])
            germanet_db.synsets.save(synset)

    print('Inserted {0} lexical relations, {1} synset relations.'.format(
        len(lex_rels), len(con_rels)))