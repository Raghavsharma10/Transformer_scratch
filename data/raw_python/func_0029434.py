def check_order(current, hit, overlap = 200):
    """
    determine if hits are sequential on model and on the
    same strand
        * if not, they should be split into different groups
    """
    prev_model = current[-1][2:4]
    prev_strand = current[-1][-2]
    hit_model = hit[2:4]
    hit_strand = hit[-2]
    # make sure they are on the same strand
    if prev_strand != hit_strand:
        return False
    # check for sequential hits on + strand
    if prev_strand == '+' and (prev_model[1] - hit_model[0] >= overlap):
        return False
    # check for sequential hits on - strand
    if prev_strand == '-' and (hit_model[1] - prev_model[0] >= overlap):
        return False
    else:
        return True