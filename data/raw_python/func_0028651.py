def _parse_table_data(self, data):
        """Parse dataset data of the original HEPData format

        :param data: header of the table to be parsed
        :raise ValueError:
        """
        header = data.split(':')

        self.current_table.data_header = header

        for i, h in enumerate(header):
            header[i] = h.strip()

        x_count = header.count('x')
        y_count = header.count('y')

        if not self.current_table.xheaders:
            raise BadFormat("*xheader line needs to appear before *data: %s" % data)

        if not self.current_table.yheaders:
            raise BadFormat("*yheader line needs to appear before *data: %s" % data)

        # use deepcopy to avoid references in yaml... may not be required, and will very probably be refactored
        # TODO - is this appropriate behavior, or are references in YAML files acceptable, they are certainly less human readable
        self.current_table.data = {'independent_variables': [{'header': self.current_table.xheaders[i] if i < len(self.current_table.xheaders) else copy.deepcopy(self.current_table.xheaders[-1]),
                                                              'values': []} for i in range(x_count)],
                                      'dependent_variables': [{'header': self.current_table.yheaders[i] if i < len(self.current_table.yheaders) else copy.deepcopy(self.current_table.yheaders[-1]),
                                                               'qualifiers': [self.current_table.qualifiers[j][i] if i < len(self.current_table.qualifiers[j]) else copy.deepcopy(self.current_table.qualifiers[j][-1]) for j in range(len(self.current_table.qualifiers)) ],
                                                               'values': []} for i in range(y_count)]}

        xy_mapping = []

        current_x_count = 0
        current_y_count = 0

        for h in header:
            if h == 'x':
                xy_mapping.append(current_x_count)
                current_x_count += 1
            if h == 'y':
                xy_mapping.append(current_y_count)
                current_y_count += 1

        last_index = self.current_file.tell()
        line = self._strip_comments(self.current_file.readline())

        while line and not line.startswith('*'):
            data_entry_elements = line.split(';')[:-1] # split and also strip newline character at the end

            if len(data_entry_elements) == len(header):
            # this is kind of a big stretch... I assume that x is always first
                for i, h in enumerate(header):
                    single_element = data_entry_elements[i].strip()

                    # number patterns copied from old subs.pl parsing script
                    pmnum1 = '[-+]?[\d]+\.?[\d]*'
                    pmnum2 = '[-+]?\.[\d]+'
                    pmnum3 = '[-+]?[\d]+\.?[\d]*\s*[eE]+\s*[+-]?\s*[\d]+'
                    pmnum = '(' + pmnum1 + '|' + pmnum2 + '|' + pmnum3 + ')'

                    # implement same regular expression matching as in old subs.pl parsing script

                    if h == 'x': # independent variables

                        r = re.search('^(?P<value>' + pmnum + ')$', single_element)
                        if r: # "value"
                            single_element = {'value': r.group('value')}
                        else:
                            r = re.search('^(?P<value>' + pmnum + ')\s*\(\s*BIN\s*=\s*(?P<low>' + pmnum + \
                                          ')\s+TO\s+(?P<high>' + pmnum + ')\s*\)$', single_element)
                            if r: # "value (BIN=low TO high)"
                                single_element = {'value': float(r.group('value')),
                                                  'low': float(r.group('low')), 'high': float(r.group('high'))}
                            else:
                                r = re.search('^(?P<low>' + pmnum + ')\s+TO\s+(?P<high>' + pmnum + ')$',
                                              single_element)
                                if r: # "low TO high"
                                    single_element = {'low': float(r.group('low')), 'high': float(r.group('high'))}
                                else: # everything else: don't try to convert to float
                                    single_element = {'value': single_element}

                        # TO DO: subs.pl also parses other formats such as "low high", "value low high" (sorted),
                        # "value +- err", and "value -err_m, +err_p".  Do we need to support these formats here?
                        # Probably not: unsupported formats will just be written as a text string.

                        self.current_table.data['independent_variables'][xy_mapping[i]]['values'].append(single_element)

                        # extract energy if SQRT(S) is one of the 'x' variables
                        xheader = self.current_table.data['independent_variables'][xy_mapping[i]]['header']
                        if xheader['name'].startswith('SQRT(S)') and lower(xheader['units']) in ('gev'):
                            for energy in single_element.values():
                                try:
                                    energy = float(energy)
                                    self.set_of_energies.add(energy)
                                except:
                                    pass

                    elif h == 'y': # dependent variable

                        pmnum_pct = pmnum + '(\s*PCT)?' # errors can possibly be given as percentages

                        r = re.search('^(?P<value>' + pmnum + ')\s+(?P<err_p>' + pmnum_pct + '|-)\s*,\s*(?P<err_m>' +
                                      pmnum_pct + '|-)\s*(?P<err_sys>\(\s*DSYS=[^()]+\s*\))?$', single_element)
                        element = {'errors': []}
                        if r: # asymmetric first error
                            element['value'] = r.group('value').strip()
                            err_p = r.group('err_p').strip().lstrip('+')
                            if err_p == '-': err_p = '' # represent missing error as '-' in oldhepdata format
                            err_p = err_p[:-3].strip() + '%' if err_p[-3:] == 'PCT' else err_p
                            err_m = r.group('err_m').strip().lstrip('+')
                            if err_m == '-': err_m = '' # represent missing error as '-' in oldhepdata format
                            err_m = err_m[:-3].strip() + '%' if err_m[-3:] == 'PCT' else err_m
                            if err_p and err_m and err_p[-1] != '%' and err_m[-1] == '%':
                                err_p = err_p + '%'
                            if not err_p and not err_m:
                                raise ValueError("Both asymmetric errors cannot be '-': %s" % line)
                            if r.group('err_sys'):
                                element['errors'] += [{'label': 'stat', 'asymerror': {'plus': err_p, 'minus': err_m}}]
                            else:
                                element['errors'] += [{'asymerror': {'plus': err_p, 'minus': err_m}}]

                        else:
                            r = re.search('^(?P<value>' + pmnum + ')\s*(\+-\s*(?P<error>' +
                                          pmnum_pct + '))?\s*(?P<err_sys>\(\s*DSYS=[^()]+\s*\))?$', single_element)
                            if r: # symmetric first error
                                element['value'] = r.group('value').strip()
                                if r.group('error'):
                                    error = r.group('error').strip().lstrip('+')
                                    error = error[:-3].strip() + '%' if error[-3:] == 'PCT' else error
                                    if r.group('err_sys'):
                                        element['errors'] += [{'label': 'stat', 'symerror': error}]
                                    else:
                                        element['errors'] += [{'symerror': error}]
                            else: # everything else
                                element['value'] = single_element

                        err_sys = []
                        if r and r.group('err_sys'):
                            err_sys = r.group('err_sys').strip(' \t()').split('DSYS=')

                        for err in err_sys + self.current_table.dserrors:
                            err = err.strip(' \t,')
                            if not err:
                                continue
                            error = {}
                            label = 'sys'
                            r = re.search('^(\+-)?\s*(?P<error>' + pmnum_pct + ')\s*(\:\s*(?P<label>.+))?$', err)
                            if r: # symmetric systematic error
                                if r.group('label'):
                                    label += ',' + r.group('label')
                                error = r.group('error').strip().lstrip('+')
                                error = error[:-3].strip() + '%' if error[-3:] == 'PCT' else error
                                error = {'symerror': error}

                            else:
                                r = re.search('^(?P<err_p>' + pmnum_pct + '|-)\s*,\s*(?P<err_m>' +
                                              pmnum_pct + '|-)\s*(\:\s*(?P<label>.+))?$', err)
                                if r: # asymmetric systematic error
                                    if r.group('label'):
                                        label += ',' + r.group('label')
                                    err_p = r.group('err_p').strip().lstrip('+')
                                    if err_p == '-': err_p = '' # represent missing error as '-' in oldhepdata format
                                    err_p = err_p[:-3].strip() + '%' if err_p[-3:] == 'PCT' else err_p
                                    err_m = r.group('err_m').strip().lstrip('+')
                                    if err_m == '-': err_m = '' # represent missing error as '-' in oldhepdata format
                                    err_m = err_m[:-3].strip() + '%' if err_m[-3:] == 'PCT' else err_m
                                    if err_p and err_m and err_p[-1] != '%' and err_m[-1] == '%':
                                        err_p = err_p + '%'
                                    if not err_p and not err_m:
                                        raise ValueError("Both asymmetric errors cannot be '-': %s" % line)
                                    error = {'asymerror': {'plus': err_p, 'minus': err_m}}
                            if not r:
                                # error happened
                                raise ValueError("Error while parsing data line: %s" % line)

                            error['label'] = label
                            if element['value'] != single_element:
                                element['errors'].append(error)
                        self.current_table.data['dependent_variables'][xy_mapping[i]]['values'].append(element)

            elif data_entry_elements:
                raise BadFormat("%s data entry elements but %s expected: %s" %
                                (len(data_entry_elements), len(header), line))

            last_index = self.current_file.tell()
            l = self.current_file.readline()
            line = self._strip_comments(l)

        self.current_file.seek(last_index)

        # extract minimum and maximum from set of energies
        if self.set_of_energies:
            energy_min = min(self.set_of_energies)
            energy_max = max(self.set_of_energies)
            if energy_max > energy_min:
                energy = str(energy_min) + '-' + str(energy_max)
            else:
                energy = energy_min
            self._parse_energies(energy)

        if self.current_table.description:
            if any(word in self.current_table.description.lower() for word in ['covariance', 'correlation', 'matrix']):
                reformatted = self._reformat_matrix()