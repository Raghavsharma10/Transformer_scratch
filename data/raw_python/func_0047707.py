def nontruncating_zip(*seqs):
        """Return a list of tuples, where each tuple contains the i-th 
        element from each of the argument sequences.

        The returned list is as long as the longest argument sequence.  
        Shorter argument sequences will be represented in the output as 
        None padding elements:

            nontruncating_zip([1, 2, 3], ['a', 'b'])
            -> [(1, 'a'), (2, 'b'), (3, None)]

        """
        n_seqs = len(seqs)

        tups = []
        idx = 0
        while True:
            empties = 0
            tup = []
            for seq in seqs:
                try:
                    tup.append(seq[idx])
                except IndexError:
                    empties += 1
                    tup.append(None)
            if empties == n_seqs:
                break
            tup = tuple(tup)
            tups.append(tup)
            idx += 1

        return tups