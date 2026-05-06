def guess_input_handler(seqs, add_seq_names=False):
    """Returns the name of the input handler for seqs."""
    if isinstance(seqs, str):
        if '\n' in seqs:  # can't be a filename...
            return '_input_as_multiline_string'
        else:  # assume it was a filename
            return '_input_as_string'

    if isinstance(seqs, list) and len(seqs) and isinstance(seqs[0], tuple):
        return '_input_as_seq_id_seq_pairs'

    if add_seq_names:
        return '_input_as_seqs'

    return '_input_as_lines'