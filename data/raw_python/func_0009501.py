def _grow(growth, walls, target, i, j, steps, new_steps, res):
    '''
    fills [res] with [distance to next position where target == 1,
                      x coord.,
                      y coord. of that position in target]
     using region growth

    i,j -> pixel position
     growth -> a work array, needed to measure the distance
     steps, new_steps -> current and last positions of the region growth steps
        using this instead of looking for the right step position in [growth]
        should speed up the process
    '''

    # clean array:
    growth[:] = 0

    if target[i, j]:
        # pixel is in target
        res[0] = 1
        res[1] = i
        res[2] = j
        return

    step = 1
    s0, s1 = growth.shape
    step_len = 1
    new_step_ind = 0

    steps[new_step_ind, 0] = i
    steps[new_step_ind, 1] = j
    growth[i, j] = 1

    while True:
        for n in range(step_len):
            i, j = steps[n]
            for ii, jj in DIRECT_NEIGHBOURS:
                pi = i + ii
                pj = j + jj

                # if in image:
                if 0 <= pi < s0 and 0 <= pj < s1:
                    # is growth array is empty and there are no walls:
                        # fill growth with current step
                    if growth[pi, pj] == 0 and not walls[pi, pj]:
                        growth[pi, pj] = step
                        if target[pi, pj]:
                            # found destination
                            res[0] = 1
                            res[1] = pi
                            res[2] = pj
                            return

                        new_steps[new_step_ind, 0] = pi
                        new_steps[new_step_ind, 1] = pj
                        new_step_ind += 1

        if new_step_ind == 0:
            # couldn't populate any more because growth is full
                # and all possible steps are gone
            res[0] = 0
            return

        step += 1
        steps, new_steps = new_steps, steps
        step_len = new_step_ind
        new_step_ind = 0