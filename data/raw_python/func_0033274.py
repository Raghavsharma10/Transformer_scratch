def ids_from_seq_two_step(seq, n, max_iterations, app, core_threshold, \
    extra_threshold, lower_threshold, second_db=None):
    """Returns ids that match a seq, using a 2-tiered strategy.

    Optionally uses a second database for the second search.
    """
    #first time through: reset 'h' and 'e' to core
    #-h is the e-value threshold for including seqs in the score matrix model
    app.Parameters['-h'].on(core_threshold)
    #-e is the e-value threshold for the final blast
    app.Parameters['-e'].on(core_threshold)
    checkpoints = []
    ids = []
    last_num_ids = None
    for i in range(max_iterations):
        if checkpoints:
            app.Parameters['-R'].on(checkpoints[-1])
        curr_check = 'checkpoint_%s.chk' % i
        app.Parameters['-C'].on(curr_check)

        output = app(seq)
        #if we didn't write a checkpoint, bail out
        if not access(curr_check, F_OK):
            break
        #if we got here, we wrote a checkpoint file
        checkpoints.append(curr_check)
        result = list(output.get('BlastOut', output['StdOut']))
        output.cleanUp()
        if result:
            ids = LastProteinIds9(result,keep_values=True,filter_identity=False)
        num_ids = len(ids)
        if num_ids >= n:
            break
        if num_ids == last_num_ids:
            break
        last_num_ids = num_ids

    #if we didn't write any checkpoints, second run won't work, so return ids
    if not checkpoints:
        return ids

    #if we got too many ids and don't have a second database, return the ids we got
    if (not second_db) and num_ids >= n:
        return ids

    #second time through: reset 'h' and 'e' to get extra hits, and switch the
    #database if appropriate
    app.Parameters['-h'].on(extra_threshold)
    app.Parameters['-e'].on(lower_threshold)
    if second_db:
        app.Parameters['-d'].on(second_db)
    for i in range(max_iterations): #will always have last_check if we get here
        app.Parameters['-R'].on(checkpoints[-1])
        curr_check = 'checkpoint_b_%s.chk' % i
        app.Parameters['-C'].on(curr_check)
        output = app(seq)
        #bail out if we couldn't write a checkpoint
        if not access(curr_check, F_OK):
            break
        #if we got here, the checkpoint worked
        checkpoints.append(curr_check)
        result = list(output.get('BlastOut', output['StdOut']))
        if result:
            ids = LastProteinIds9(result,keep_values=True,filter_identity=False)
        num_ids = len(ids)
        if num_ids >= n:
            break
        if num_ids == last_num_ids:
            break
        last_num_ids = num_ids
    #return the ids we got. may not be as many as we wanted.
    for c in checkpoints:
        remove(c)
    return ids