def get_ss_regions(assembly, ss_types):
    """Returns an Assembly containing Polymers for each region of structure.

    Parameters
    ----------
    assembly : ampal.Assembly
        `Assembly` object to be searched secondary structure regions.
    ss_types : list
        List of secondary structure tags to be separate i.e. ['H']
        would return helices, ['H', 'E'] would return helices
        and strands.

    Returns
    -------
    fragments : Assembly
        `Assembly` containing a `Polymer` for each region of specified
        secondary structure.
    """
    if not any(map(lambda x: 'ss_regions' in x.tags, assembly)):
        raise ValueError(
            'This assembly does not have any tagged secondary structure '
            'regions. Use `ampal.dssp.tag_dssp_data` to add the tags.'
        )
    fragments = Assembly()
    for polypeptide in assembly:
        if 'ss_regions' in polypeptide.tags:
            for start, end, ss_type in polypeptide.tags['ss_regions']:
                if ss_type in ss_types:
                    fragment = polypeptide.get_slice_from_res_id(start, end)
                    fragments.append(fragment)
    if not fragments:
        raise ValueError('No regions matching that secondary structure type'
                         ' have been found. Use standard DSSP labels.')
    return fragments