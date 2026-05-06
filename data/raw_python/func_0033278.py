def ids_from_seqs_iterative(seqs, app, query_parser, \
    scorer=keep_everything_scorer, max_iterations=None, blast_db=None,\
    max_seqs=None, ):
    """Gets the ids from each seq, then does each additional id until all done.

    If scorer is passed in as an int, uses shotgun scorer with that # hits.
    """
    if isinstance(scorer, int):
        scorer = make_shotgun_scorer(scorer)
    seqs_to_check = list(seqs)
    checked_ids = {}
    curr_iteration = 0
    while seqs_to_check:
        unchecked_ids = {}
        #pass seqs to command
        all_output = app(seqs_to_check)
        output = all_output.get('BlastOut', all_output['StdOut'])

        for query_id, match_id, match_score in query_parser(output):
            if query_id not in checked_ids:
                checked_ids[query_id] = {}
            checked_ids[query_id][match_id] = match_score
            if match_id not in checked_ids:
                unchecked_ids[match_id] = True
        all_output.cleanUp()
        if unchecked_ids:
            seq_file = fasta_cmd_get_seqs(unchecked_ids.keys(),
                app.Parameters['-d'].Value)['StdOut']
            seqs_to_check = []
            for s in FastaCmdFinder(fasta_cmd_get_seqs(\
                unchecked_ids.keys(), app.Parameters['-d'].Value)['StdOut']):
                seqs_to_check.extend(s)
        else:
            seqs_to_check = []
        #bail out if max iterations or max seqs was defined and we've reached it
        curr_iteration += 1
        if max_iterations and (curr_iteration >= max_iterations):
            break
        if max_seqs:
            curr = scorer(checked_ids)
            if len(curr) >= max_seqs:
                return curr
    return scorer(checked_ids)