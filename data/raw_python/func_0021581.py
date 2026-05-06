def calc_plateaus(beta, edges, rel_tol=1e-4, verbose=0):
    '''Calculate the plateaus (degrees of freedom) of a graph of beta values in linear time.'''
    if not isinstance(edges, dict):
        raise Exception('Edges must be a map from each node to a list of neighbors.')
    to_check = deque(range(len(beta)))
    check_map = np.zeros(beta.shape, dtype=bool)
    check_map[np.isnan(beta)] = True
    plateaus = []

    if verbose:
        print('\tCalculating plateaus...')

    if verbose > 1:
        print('\tIndices to check {0} {1}'.format(len(to_check), check_map.shape))

    # Loop until every beta index has been checked
    while to_check:
        if verbose > 1:
            print('\t\tPlateau #{0}'.format(len(plateaus) + 1))

        # Get the next unchecked point on the grid
        idx = to_check.popleft()

        # If we already have checked this one, just pop it off
        while to_check and check_map[idx]:
            try:
                idx = to_check.popleft()
            except:
                break

        # Edge case -- If we went through all the indices without reaching an unchecked one.
        if check_map[idx]:
            break

        # Create the plateau and calculate the inclusion conditions
        cur_plateau = set([idx])
        cur_unchecked = deque([idx])
        val = beta[idx]
        min_member = val - rel_tol
        max_member = val + rel_tol

        # Check every possible boundary of the plateau
        while cur_unchecked:
            idx = cur_unchecked.popleft()
            
            # neighbors to check
            local_check = []

            # Generic graph case, get all neighbors of this node
            local_check.extend(edges[idx])

            # Check the index's unchecked neighbors
            for local_idx in local_check:
                if not check_map[local_idx] \
                    and beta[local_idx] >= min_member \
                    and beta[local_idx] <= max_member:
                        # Label this index as being checked so it's not re-checked unnecessarily
                        check_map[local_idx] = True

                        # Add it to the plateau and the list of local unchecked locations
                        cur_unchecked.append(local_idx)
                        cur_plateau.add(local_idx)

        # Track each plateau's indices
        plateaus.append((val, cur_plateau))

    # Returns the list of plateaus and their values
    return plateaus