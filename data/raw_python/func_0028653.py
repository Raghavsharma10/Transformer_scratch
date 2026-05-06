def _parse_qual(self, data):
        """Parse qual attribute of the old HEPData format

        example qual:
        *qual: RE : P P --> Z0 Z0 X

        :param data: data to be parsed
        :type data: str
        """
        list = []
        headers = data.split(':')
        name = headers[0].strip()

        name = re.split(' IN ', name, flags=re.I) # ignore case
        units = None
        if len(name) > 1:
            units = name[1].strip()
        name = name[0].strip()

        if len(headers) < 2:
            raise BadFormat("*qual line must contain a name and values: %s" % data)

        for header in headers[1:]:
            xheader = {'name': name}
            if units:
                xheader['units'] = units

            xheader['value'] = header.strip()
            list.append(xheader)

            # extract energy if SQRT(S) is one of the qualifiers
            if name.startswith('SQRT(S)') and lower(units) in ('gev'):
                energies = re.split(' TO ', xheader['value'], flags=re.I)
                for energy in energies:
                    try:
                        energy = float(energy)
                        self.set_of_energies.add(energy)
                    except:
                        pass

        self.current_table.qualifiers.append(list)