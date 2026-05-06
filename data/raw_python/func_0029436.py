def find_coordinates(hmms, bit_thresh):
    """
    find 16S rRNA gene sequence coordinates
    """
    # get coordinates from cmsearch output
    seq2hmm = parse_hmm(hmms, bit_thresh)
    seq2hmm = best_model(seq2hmm)
    group2hmm = {} # group2hmm[seq][group] = [model, strand, coordinates, matches, gaps]
    for seq, info in list(seq2hmm.items()):
        group2hmm[seq] = {}
        # info = [model, [[hit1], [hit2], ...]]
        for group_num, group in enumerate(hit_groups(info[1])):
            # group is a group of hits to a single 16S gene
            # determine matching strand based on best hit
            best = sorted(group, reverse = True, key = itemgetter(-1))[0]
            strand = best[5]
            coordinates = [i[0] for i in group] + [i[1] for i in group]
            coordinates = [min(coordinates), max(coordinates), strand]
            # make sure all hits are to the same strand
            matches = [i for i in group if i[5] == strand]
            # gaps = [[gstart, gend], [gstart2, gend2]]
            gaps = check_gaps(matches)
            group2hmm[seq][group_num] = [info[0], strand, coordinates, matches, gaps]
    return group2hmm