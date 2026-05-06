def check_overlap(current, hit, overlap = 200):
    """
    determine if sequence has already hit the same part of the model,
    indicating that this hit is for another 16S rRNA gene
    """
    for prev in current:
        p_coords = prev[2:4]
        coords = hit[2:4]
        if get_overlap(coords, p_coords) >= overlap:
            return True
    return False