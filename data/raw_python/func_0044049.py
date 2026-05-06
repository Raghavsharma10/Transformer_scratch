def get_atom_type_symbol(cls,calc,**kwargs):
        """
        Returns a list of atom types. Each atom site MUST occur only
        once in this list. List MUST be sorted.
        """
        parameters = calc.out.output
        dictionary = parameters.get_dict()
        if 'basis_set' not in dictionary.keys():
            return None
        return sorted(dictionary['basis_set'].keys())