def ids_from_seq_lower_threshold(seq, n, max_iterations, app, core_threshold, \
    lower_threshold, step=100):
    """Returns ids that match a seq, decreasing the sensitivity."""
    last_num_ids = None
    checkpoints = []
    cp_name_base = make_unique_str()

    # cache ides for each iteration
    # store { iteration_num:(core_threshold, [list of matching ids]) }
    all_ids = {}
    try:
        i=0
        while 1:
            #-h is the e-value threshold for inclusion in the score matrix model
            app.Parameters['-h'].on(core_threshold)
            app.Parameters['-e'].on(core_threshold)
            if core_threshold > lower_threshold:
                raise ThresholdFound
            if checkpoints:
                #-R restarts from a previously stored file
                app.Parameters['-R'].on(checkpoints[-1])
            #store the score model from this iteration
            curr_check = 'checkpoint_' + cp_name_base + '_' + str(i) + \
                    '.chk'
            app.Parameters['-C'].on(curr_check)
            output = app(seq)
            result = list(output.get('BlastOut', output['StdOut']))
            #sometimes fails on first try -- don't know why, but this seems
            #to fix problem
            while not result:
                output = app(seq)
                result = list(output.get('BlastOut', output['StdOut']))

            ids = LastProteinIds9(result,keep_values=True,filter_identity=False)
            output.cleanUp()
            all_ids[i + 1] = (core_threshold, copy(ids))
            if not access(curr_check, F_OK):
                raise ThresholdFound
            checkpoints.append(curr_check)
            num_ids = len(ids)
            if num_ids >= n:
                raise ThresholdFound
            last_num_ids = num_ids
            core_threshold *= step
            if i >= max_iterations - 1: #because max_iterations is 1-based
                raise ThresholdFound
            i += 1
    except ThresholdFound:
        for c in checkpoints:
            remove(c)
        #turn app.Parameters['-R'] off so that for the next file it does not
        #try and read in a checkpoint file that is not there
        app.Parameters['-R'].off()
        return ids, i + 1, all_ids