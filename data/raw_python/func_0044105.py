def proc_line_coordinate(self, line):
        """Extracts data from columns in ATOM/HETATM record."""
        at_type = line[0:6].strip()  # 0
        at_ser = int(line[6:11].strip())  # 1
        at_name = line[12:16].strip()  # 2
        alt_loc = line[16].strip()  # 3
        res_name = line[17:20].strip()  # 4
        chain_id = line[21].strip()  # 5
        res_seq = int(line[22:26].strip())  # 6
        i_code = line[26].strip()  # 7
        x = float(line[30:38].strip())  # 8
        y = float(line[38:46].strip())  # 9
        z = float(line[46:54].strip())  # 10
        occupancy = float(line[54:60].strip())  # 11
        temp_factor = float(line[60:66].strip())  # 12
        element = line[76:78].strip()  # 13
        charge = line[78:80].strip()  # 14
        if at_name not in PDB_ATOM_COLUMNS:
            PDB_ATOM_COLUMNS[at_name] = line[12:16]
            self.new_labels = True
        return (at_type, at_ser, at_name, alt_loc, res_name, chain_id, res_seq,
                i_code, x, y, z, occupancy, temp_factor, element, charge)