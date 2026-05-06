def parse_meta_data(self, line):
        """Parse a vcf metadataline"""
        line = line.rstrip()
        logger.debug("Parsing metadata line:{0}".format(line))
        line_info = line[2:].split('=')
        match = False

        if line_info[0] == 'fileformat':
            logger.debug("Parsing fileformat")
            try:
                self.fileformat = line_info[1]
                logger.debug("Found fileformat {0}".format(self.fileformat))
            except IndexError:
                raise SyntaxError("fileformat must have a value")

        elif line_info[0] == 'INFO':
            match = self.info_pattern.match(line)
            if not match:
                raise SyntaxError("One of the INFO lines is malformed:{0}".format(line))

            matches = [
                match.group('id'), match.group('number'),
                match.group('type'), match.group('desc')
            ]

            # extra_info is a dictionary to check the metadata about the INFO values:
            self.extra_info[matches[0]] = dict(
                zip(self.header_keys['info'][1:], matches[1:])
            )

            info_line = dict(list(zip(self.header_keys['info'],matches)))

            if len(info_line['Description'].split('Format:')) > 1:
                info_line['Format'] = [
                    info.strip() for info in info_line['Description'].split('Format:')
                ][-1]
            self.info_lines.append(info_line)

            # Store the vep columns:
            if info_line['ID'] == 'CSQ':
                self.vep_columns = info_line.get('Format', '').split('|')

            if info_line['ID'] == 'ANN':
                self.snpeff_columns = [
                    annotation.strip("' ") for annotation in
                    info_line.get('Description', '').split(':')[-1].split('|')]

            self.info_dict[match.group('id')] = line

        elif line_info[0] == 'FILTER':
            match = self.filter_pattern.match(line)
            if not match:
                raise SyntaxError("One of the FILTER lines is malformed: {0}".format(line))
            matches = [match.group('id'), match.group('desc')]
            self.filter_lines.append(dict(
                list(zip(self.header_keys['filt'],matches)))
            )
            self.filter_dict[match.group('id')] = line

        elif line_info[0] == 'contig':
            match = self.contig_pattern.match(line)
            if not match:
                print()
                raise SyntaxError("One of the contig lines is malformed: {0}".format(line))

            matches = [match.group('id'), match.group('length')]
            self.contig_lines.append(dict(
                list(zip(self.header_keys['contig'],matches)))
            )
            self.contig_dict[match.group('id')] = line

        elif line_info[0] == 'FORMAT':
            match = self.format_pattern.match(line)
            if not match:
                raise SyntaxError("One of the FORMAT lines is malformed: {0}".format(line))

            matches = [
                match.group('id'), match.group('number'),
                match.group('type'), match.group('desc')
            ]
            self.format_lines.append(dict(
                list(zip(self.header_keys['form'],matches)))
            )
            self.format_dict[match.group('id')] = line

        elif line_info[0] == 'ALT':
            match = self.alt_pattern.match(line)
            if not match:
                raise SyntaxError("One of the ALT lines is malformed: {0}".format(line))

            matches = [match.group('id'), match.group('desc')]
            self.alt_lines.append(dict(
                list(zip(self.header_keys['alt'],matches)))
            )
            self.alt_dict[match.group('id')] = line

        else:
            match = self.meta_pattern.match(line)
            if not match:
                raise SyntaxError("One of the meta data lines is malformed: {0}".format(line))

            self.other_lines.append({match.group('key'): match.group('val')})
            self.other_dict[match.group('key')] = line