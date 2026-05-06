def csview(self, view=False):
        """View chemical shift values organized by amino acid residue.

        :param view: Open in default image viewer or save file in current working directory quietly.
        :type view: :py:obj:`True` or :py:obj:`False`
        :return: None
        :rtype: :py:obj:`None`
        """
        for starfile in fileio.read_files(self.from_path):
            chains = starfile.chem_shifts_by_residue(amino_acids=self.amino_acids,
                                                     atoms=self.atoms,
                                                     amino_acids_and_atoms=self.amino_acids_and_atoms,
                                                     nmrstar_version=self.nmrstar_version)

            for idx, chemshifts_dict in enumerate(chains):
                nodes = []
                edges = []

                for seq_id in chemshifts_dict:
                    aaname = "{}_{}".format(chemshifts_dict[seq_id]["AA3Code"], seq_id)
                    label = '"{{{}|{}}}"'.format(seq_id, chemshifts_dict[seq_id]["AA3Code"])
                    color = 8
                    aanode_entry = "            {} [label={}, fillcolor={}]".format(aaname, label, color)
                    nodes.append(aanode_entry)
                    currnodename = aaname

                    for atom_type in chemshifts_dict[seq_id]:
                        if atom_type in ["AA3Code", "Seq_ID"]:
                            continue
                        else:
                            atname = "{}_{}".format(aaname, atom_type)
                            label = '"{{{}|{}}}"'.format(atom_type, chemshifts_dict[seq_id][atom_type])
                            if atom_type.startswith("H"):
                                color = 4
                            elif atom_type.startswith("C"):
                                color = 6
                            elif atom_type.startswith("N"):
                                color = 10
                            else:
                                color = 8
                            atnode_entry = "{} [label={}, fillcolor={}]".format(atname, label, color)
                            nextnodename = atname
                            nodes.append(atnode_entry)
                            edges.append("{} -> {}".format(currnodename, nextnodename))
                            currnodename = nextnodename

                if self.filename is None:
                    filename = "{}_{}".format(starfile.id, idx)
                else:
                    filename = "{}_{}".format(self.filename, idx)

                src = Source(self.dot_template.format("\n".join(nodes), "\n".join(edges)), format=self.csview_format)
                src.render(filename=filename, view=view)