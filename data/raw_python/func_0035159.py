def FunctionalGroupColorMapping(maptype='jet', reverse=False):
    """Maps amino-acid functional groups to colors.
    Currently does not use the keyword arguments for *maptype*
    or *reverse* but accepts these arguments to be consistent
    with the other mapping functions, which all get called with 
    these arguments."""

    small_color = '#f76ab4'
    nucleophilic_color = '#ff7f00'
    hydrophobic_color = '#12ab0d'
    aromatic_color = '#84380b'
    acidic_color = '#e41a1c'
    amide_color = '#972aa8'
    basic_color = '#3c58e5'

    mapping_d = {'G':small_color, 'A':small_color,
                 'S':nucleophilic_color, 'T':nucleophilic_color, 'C':nucleophilic_color,
                 'V':hydrophobic_color, 'L':hydrophobic_color, 'I':hydrophobic_color, 'M':hydrophobic_color, 'P':hydrophobic_color,
                 'F':aromatic_color, 'Y':aromatic_color, 'W':aromatic_color,
                 'D':acidic_color, 'E':acidic_color,
                 'H':basic_color, 'K':basic_color, 'R':basic_color,
                 'N':amide_color, 'Q':amide_color,
                 '*':'#000000'}
    return (None, mapping_d, None)