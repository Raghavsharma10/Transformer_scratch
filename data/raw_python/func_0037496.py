def chem_shifts_by_residue(self, amino_acids=None, atoms=None, amino_acids_and_atoms=None, nmrstar_version="3"):
        """Organize chemical shifts by amino acid residue.

        :param list amino_acids: List of amino acids three-letter codes.
        :param list atoms: List of BMRB atom type codes.
        :param dict amino_acids_and_atoms: Amino acid and its atoms key-value pairs. 
        :param str nmrstar_version: Version of NMR-STAR format to use for look up chemical shifts loop.
        :return: List of OrderedDict per each chain
        :rtype: :py:class:`list` of :py:class:`collections.OrderedDict`
        """
        if (amino_acids_and_atoms and amino_acids) or (amino_acids_and_atoms and atoms):
            raise ValueError('"amino_acids_and_atoms" parameter cannot be used simultaneously with '
                             '"amino_acids" and "atoms" parameters, one or another must be provided.')

        chemshifts_loop = NMRSTAR_CONSTANTS[nmrstar_version]["chemshifts_loop"]
        aminoacid_seq_id = NMRSTAR_CONSTANTS[nmrstar_version]["aminoacid_seq_id"]
        aminoacid_code = NMRSTAR_CONSTANTS[nmrstar_version]["aminoacid_code"]
        atom_code = NMRSTAR_CONSTANTS[nmrstar_version]["atom_code"]
        chemshift_value = NMRSTAR_CONSTANTS[nmrstar_version]["chemshift_value"]

        chains = []
        for saveframe in self:
            if saveframe == u"data" or saveframe.startswith(u"comment"):
                continue
            else:
                for ind in self[saveframe].keys():
                    if ind.startswith(u"loop_"):
                        if list(self[saveframe][ind][0]) == chemshifts_loop:
                            chem_shifts_dict = OrderedDict()
                            for entry in self[saveframe][ind][1]:
                                residue_id = entry[aminoacid_seq_id]
                                chem_shifts_dict.setdefault(residue_id, OrderedDict())
                                chem_shifts_dict[residue_id][u"AA3Code"] = entry[aminoacid_code]
                                chem_shifts_dict[residue_id][u"Seq_ID"] = residue_id
                                chem_shifts_dict[residue_id][entry[atom_code]] = entry[chemshift_value]
                            chains.append(chem_shifts_dict)

        if amino_acids_and_atoms:
            for chem_shifts_dict in chains:
                for aa_dict in list(chem_shifts_dict.values()):
                    if aa_dict[u"AA3Code"].upper() not in list(amino_acids_and_atoms.keys()):
                        chem_shifts_dict.pop(aa_dict[u"Seq_ID"])
                    else:
                        for resonance in list(aa_dict.keys()):
                            if resonance in (u"AA3Code", u"Seq_ID") or resonance.upper() in amino_acids_and_atoms[aa_dict[u"AA3Code"]]:
                                continue
                            else:
                                aa_dict.pop(resonance)
        else:
            if amino_acids:
                for chem_shifts_dict in chains:
                    for aa_dict in list(chem_shifts_dict.values()):
                        if aa_dict[u"AA3Code"].upper() not in amino_acids:
                            chem_shifts_dict.pop(aa_dict[u"Seq_ID"])

            if atoms:
                for chem_shifts_dict in chains:
                    for aa_dict in chem_shifts_dict.values():
                        for resonance in list(aa_dict.keys()):
                            if resonance in (u"AA3Code", u"Seq_ID") or resonance.upper() in atoms:
                                continue
                            else:
                                aa_dict.pop(resonance)
        return chains