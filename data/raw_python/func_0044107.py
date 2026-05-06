def check_for_non_canonical(residue):
        """Checks to see if the residue is non-canonical."""
        res_label = list(residue[0])[0][2]
        atom_labels = {x[2] for x in itertools.chain(
            *residue[1].values())}  # Used to find unnatural aas
        if (all(x in atom_labels for x in ['N', 'CA', 'C', 'O'])) and (
                len(res_label) == 3):
            return Residue, True
        return None