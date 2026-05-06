def tag_dssp_data(assembly, loop_assignments=(' ', 'B', 'S', 'T')):
    """Adds output data from DSSP to an Assembly.

    A dictionary will be added to the `tags` dictionary of each
    residue called `dssp_data`, which contains the secondary
    structure definition, solvent accessibility phi and psi values
    from DSSP. A list of regions of continuous secondary assignments
    will also be added to each `Polypeptide`.

    The tags are added in place, so nothing is returned from this
    function.

    Parameters
    ----------
    assembly : ampal.Assembly
        An Assembly containing some protein.
    loop_assignments : tuple or list
        A tuple containing the DSSP secondary structure identifiers to
        that are classed as loop regions.
    """
    dssp_out = run_dssp(assembly.pdb, path=False)
    dssp_data = extract_all_ss_dssp(dssp_out, path=False)
    for record in dssp_data:
        rnum, sstype, chid, _, phi, psi, sacc = record
        assembly[chid][str(rnum)].tags['dssp_data'] = {
            'ss_definition': sstype,
            'solvent_accessibility': sacc,
            'phi': phi,
            'psi': psi
        }
    ss_regions = find_ss_regions(dssp_data, loop_assignments)
    for region in ss_regions:
        chain = region[0][2]
        ss_type = ' ' if region[0][1] in loop_assignments else region[0][1]
        first_residue = str(region[0][0])
        last_residue = str(region[-1][0])
        if not 'ss_regions' in assembly[chain].tags:
            assembly[chain].tags['ss_regions'] = []
        assembly[chain].tags['ss_regions'].append(
            (first_residue, last_residue, ss_type))
    return