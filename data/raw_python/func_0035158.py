def ChargeColorMapping(maptype='jet', reverse=False):
    """Maps amino-acid charge at neutral pH to colors. 
    Currently does not use the keyword arguments for *maptype*
    or *reverse* but accepts these arguments to be consistent
    with KyteDoolittleColorMapping and MWColorMapping for now."""

    pos_color = '#FF0000'
    neg_color = '#0000FF'
    neut_color = '#000000'

    mapping_d = {'A':neut_color,'R':pos_color,'N':neut_color,\
                 'D':neg_color,'C':neut_color,'Q':neut_color,\
                 'E':neg_color,'G':neut_color,'H':pos_color,\
                 'I':neut_color,'L':neut_color,'K':pos_color,\
                 'M':neut_color,'F':neut_color,'P':neut_color,\
                 'S':neut_color,'T':neut_color,'W':neut_color,\
                 'Y':neut_color,'V':neut_color}

    return (None, mapping_d, None)