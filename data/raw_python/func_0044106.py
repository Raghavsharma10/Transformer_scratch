def gen_states(self, monomer_data, parent):
        """Generates the `states` dictionary for a `Monomer`.

        monomer_data : list
            A list of atom data parsed from the input PDB.
        parent : ampal.Monomer
            `Monomer` used to assign `parent` on created
            `Atoms`.
        """
        states = {}
        for atoms in monomer_data:
            for atom in atoms:
                state = 'A' if not atom[3] else atom[3]
                if state not in states:
                    states[state] = OrderedDict()
                states[state][atom[2]] = Atom(
                    tuple(atom[8:11]), atom[13], atom_id=atom[1],
                    res_label=atom[2], occupancy=atom[11], bfactor=atom[12],
                    charge=atom[14], state=state, parent=parent)

        # This code is to check if there are alternate states and populate any
        # both states with the full complement of atoms
        states_len = [(k, len(x)) for k, x in states.items()]
        if (len(states) > 1) and (len(set([x[1] for x in states_len])) > 1):
            for t_state, t_state_d in states.items():
                new_s_dict = OrderedDict()
                for k, v in states[sorted(states_len,
                                          key=lambda x: x[0])[0][0]].items():
                    if k not in t_state_d:
                        c_atom = Atom(
                            v._vector, v.element, atom_id=v.id,
                            res_label=v.res_label,
                            occupancy=v.tags['occupancy'],
                            bfactor=v.tags['bfactor'], charge=v.tags['charge'],
                            state=t_state[0], parent=v.parent)
                        new_s_dict[k] = c_atom
                    else:
                        new_s_dict[k] = t_state_d[k]
                states[t_state] = new_s_dict
        return states