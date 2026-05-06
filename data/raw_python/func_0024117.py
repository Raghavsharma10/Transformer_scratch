def get_pID(read):
    """Return the percent identity of a read.

    based on the NM tag if present,
    if not calculate from MD tag and CIGAR string

    read.query_alignment_length can be zero in the case of ultra long reads aligned with minimap2 -L
    """
    try:
        return 100 * (1 - read.get_tag("NM") / read.query_alignment_length)
    except KeyError:
        try:
            return 100 * (1 - (parse_MD(read.get_tag("MD")) + parse_CIGAR(read.cigartuples))
                          / read.query_alignment_length)
        except KeyError:
            return None
    except ZeroDivisionError:
        return None