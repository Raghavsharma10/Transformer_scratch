def find(find_seq, elem_list):
        """
        Return the first position in elem_list where find_seq starts
        """
        seq_pos = 0
        for index, elem in enumerate(elem_list):
            if Sequence.match(elem, find_seq[seq_pos]):
                seq_pos += 1
                if seq_pos == len(find_seq):  # found matching sequence
                    return index - seq_pos + 1
            else:  # exited sequence
                seq_pos = 0
        raise LookupError('Failed to find sequence in elem_list')