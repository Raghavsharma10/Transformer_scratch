def write_pdb(residues, chain_id=' ', alt_states=False, strip_states=False):
    """Writes a pdb file for a list of residues.

    Parameters
    ----------
    residues : list
        List of Residue objects.
    chain_id : str
        String of the chain id, defaults to ' '.
    alt_states : bool, optional
        If true, include all occupancy states of residues, else outputs primary state only.
    strip_states : bool, optional
        If true, remove all state labels from residues. Only use with alt_states false.

    Returns
    -------
    pdb_str : str
        String of the PDB file.
    """
    pdb_atom_col_dict = PDB_ATOM_COLUMNS
    out_pdb = []
    if len(str(chain_id)) > 1:
        poly_id = ' '
    else:
        poly_id = str(chain_id)
    for monomer in residues:
        if (len(monomer.states) > 1) and alt_states and not strip_states:
            atom_list = itertools.chain(
                *[x[1].items() for x in sorted(monomer.states.items())])
        else:
            atom_list = monomer.atoms.items()
        if 'chain_id' in monomer.tags:
            poly_id = monomer.tags['chain_id']
        for atom_t, atom in atom_list:
            if strip_states:
                state_label = ' '
            elif (atom.tags['state'] == 'A') and (len(monomer.states) == 1):
                state_label = ' '
            else:
                state_label = atom.tags['state']
            atom_data = {
                'atom_number': '{:>5}'.format(cap(atom.id, 5)),
                'atom_name': '{:<4}'.format(cap(pdb_atom_col_dict[atom_t], 4)),
                'alt_loc_ind': '{:<1}'.format(cap(state_label, 1)),
                'residue_type': '{:<3}'.format(cap(monomer.mol_code, 3)),
                'chain_id': '{:<1}'.format(cap(poly_id, 1)),
                'res_num': '{:>4}'.format(cap(monomer.id, 4)),
                'icode': '{:<1}'.format(cap(monomer.insertion_code, 1)),
                'coord_str': '{0:>8.3f}{1:>8.3f}{2:>8.3f}'.format(
                    *[x for x in atom]),
                'occupancy': '{:>6.2f}'.format(atom.tags['occupancy']),
                'temp_factor': '{:>6.2f}'.format(atom.tags['bfactor']),
                'element': '{:>2}'.format(cap(atom.element, 2)),
                'charge': '{:<2}'.format(cap(atom.tags['charge'], 2))
            }
            if monomer.is_hetero:
                pdb_line_template = (
                    'HETATM{atom_number} {atom_name}{alt_loc_ind}{residue_type}'
                    ' {chain_id}{res_num}{icode}   {coord_str}{occupancy}'
                    '{temp_factor}          {element}{charge}\n'
                )
            else:
                pdb_line_template = (
                    'ATOM  {atom_number} {atom_name}{alt_loc_ind}{residue_type}'
                    ' {chain_id}{res_num}{icode}   {coord_str}{occupancy}'
                    '{temp_factor}          {element}{charge}\n'
                )
            out_pdb.append(pdb_line_template.format(**atom_data))
    return ''.join(out_pdb)