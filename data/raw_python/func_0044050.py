def get_atom_type_basisset(cls,calc,**kwargs):
        """
        Returns a list of basisset names for each atom type. The list
        order MUST be the same as of get_atom_type_symbol().
        """
        parameters = calc.out.output
        dictionary = parameters.get_dict()
        if 'basis_set' not in dictionary.keys():
            return None
        return [dictionary['basis_set'][x]['description']
                for x in cls.get_atom_type_symbol(calc,**kwargs)]