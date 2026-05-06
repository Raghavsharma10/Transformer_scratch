def _read(self, filename, field_labels=None):
        """Read all the requested fields

           Arguments:
            | ``filename``  --  the filename of the FCHK file
            | ``field_labels``  --  when given, only these fields are read
        """
        # if fields is None, all fields are read
        def read_field(f):
            """Read a single field"""
            datatype = None
            while datatype is None:
                # find a sane header line
                line = f.readline()
                if line == "":
                    return False

                label = line[:43].strip()
                if field_labels is not None:
                    if len(field_labels) == 0:
                        return False
                    elif label not in field_labels:
                        return True
                    else:
                        field_labels.discard(label)
                line = line[43:]
                words = line.split()
                if len(words) == 0:
                    return True

                if words[0] == 'I':
                    datatype = int
                    unreadable = 0
                elif words[0] == 'R':
                    datatype = float
                    unreadable = np.nan

            if len(words) == 2:
                try:
                    value = datatype(words[1])
                except ValueError:
                    return True
            elif len(words) == 3:
                if words[1] != "N=":
                    raise FileFormatError("Unexpected line in formatted checkpoint file %s\n%s" % (filename, line[:-1]))
                length = int(words[2])
                value = np.zeros(length, datatype)
                counter = 0
                try:
                    while counter < length:
                        line = f.readline()
                        if line == "":
                            raise FileFormatError("Unexpected end of formatted checkpoint file %s" % filename)
                        for word in line.split():
                            try:
                                value[counter] = datatype(word)
                            except (ValueError, OverflowError) as e:
                                print('WARNING: could not interpret word while reading %s: %s' % (word, self.filename))
                                if self.ignore_errors:
                                    value[counter] = unreadable
                                else:
                                    raise
                            counter += 1
                except ValueError:
                    return True
            else:
                raise FileFormatError("Unexpected line in formatted checkpoint file %s\n%s" % (filename, line[:-1]))

            self.fields[label] = value
            return True

        self.fields = {}
        with open(filename, 'r') as f:
            self.title = f.readline()[:-1].strip()
            words = f.readline().split()
            if len(words) == 3:
                self.command, self.lot, self.basis = words
            elif len(words) == 2:
                self.command, self.lot = words
            else:
                raise FileFormatError('The second line of the FCHK file should contain two or three words.')

            while read_field(f):
                pass