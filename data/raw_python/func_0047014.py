def get_three_frame_orfs(sequence, starts=None, stops=None):
    """Find ORF's in frames 1, 2 and 3 for the given sequence.

    Positions returned are 1-based (not 0)

    Return format [{'start': start_position, 'stop': stop_position, 'sequence': sequence}, ]

    Keyword arguments:
    sequence -- sequence for the transcript
    starts -- List of codons to be considered as start (Default: ['ATG'])
    stops -- List of codons to be considered as stop (Default: ['TAG', 'TGA', 'TAA'])

    """
    if not starts:
        starts = ['ATG']

    if not stops:
        stops = ['TAG', 'TGA', 'TAA']

    # Find ORFs in 3 frames
    orfs = []
    for frame in range(3):
        start_codon = None
        orf = ''
        for position in range(frame, len(sequence), 3):
            codon = sequence[position:position + 3]
            if codon in starts:
                # We have found a start already, so add codon to orf and
                # continue. This is an internal MET
                if start_codon is not None:
                    orf += codon
                    continue

                # New orf start
                start_codon = position
                orf = codon
            else:
                # if sequence starts with ATG, start_codon will be 0
                if start_codon is None:
                    # We haven't found a start codon yet
                    continue
                orf += codon
                if codon in stops:
                    # orfs[start_codon + 1] = orf
                    orfs.append({'start': start_codon + 1, 'stop': position + 3, 'sequence': orf})

                    # Reset
                    start_codon = None
                    orf = ''
    return orfs