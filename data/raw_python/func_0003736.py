def _load_bond_data(self):
        """Load the bond data from the given file

           It's assumed that the uncommented lines in the data file have the
           following format:
           symbol1 symbol2 number1 number2 bond_length_single_a bond_length_double_a bond_length_triple_a bond_length_single_b bond_length_double_b bond_length_triple_b ..."
           where a, b, ... stand for different sources.
        """

        def read_units(unit_names):
            """convert unit_names into conversion factors"""
            tmp = {
                "A": units.angstrom,
                "pm": units.picometer,
                "nm": units.nanometer,
            }
            return [tmp[unit_name] for unit_name in unit_names]

        def read_length(BOND_TYPE, words, col):
            """Read the bondlengths from a single line in the data file"""
            nlow = int(words[2])
            nhigh = int(words[3])
            for i, conversion in zip(range((len(words) - 4) // 3), conversions):
                word = words[col + 3 + i*3]
                if word != 'NA':
                    self.lengths[BOND_TYPE][frozenset([nlow, nhigh])] = float(word)*conversion
                    return

        with pkg_resources.resource_stream(__name__, 'data/bonds.csv') as f:
            for line in f:
                words = line.decode('utf-8').split()
                if (len(words) > 0) and (words[0][0] != "#"):
                    if words[0] == "unit":
                        conversions = read_units(words[1:])
                    else:
                        read_length(BOND_SINGLE, words, 1)
                        read_length(BOND_DOUBLE, words, 2)
                        read_length(BOND_TRIPLE, words, 3)