def load(self, filename, subset=None):
        """Load data into the registered fields

           Argument:
            | ``filename``  --  the filename to read from

           Optional argument:
            | ``subset``  --  a list of field names that are read from the file.
                              If not given, all data is read from the file.
        """
        with open(filename, "r") as f:
            name = None
            num_names = 0

            while True:
                # read a header line
                line = f.readline()
                if len(line) == 0:
                    break

                # process the header line
                words = line.split()
                name = words[0]
                attr = self._fields.get(name)
                if attr is None:
                    raise FileFormatError("Wrong header: unknown field %s" % name)

                if not words[1].startswith("kind="):
                    raise FileFormatError("Malformatted array header line. (kind)")
                kind = words[1][5:]
                expected_kind = attr.get_kind(attr.get())
                if kind != expected_kind:
                    raise FileFormatError("Wrong header: kind of field %s does not match. Got %s, expected %s" % (name, kind, expected_kind))

                skip = ((subset is not None) and (name not in subset))

                print(words)
                if (words[2].startswith("shape=(") and words[2].endswith(")")):
                    if not isinstance(attr, ArrayAttr):
                        raise FileFormatError("field '%s' is not an array." % name)
                    shape = words[2][7:-1]
                    if shape[-1] == ', ':
                        shape = shape[:-1]
                    try:
                        shape = tuple(int(word) for word in shape.split(","))
                    except ValueError:
                        raise FileFormatError("Malformatted array header. (shape)")
                    expected_shape = attr.get().shape
                    if shape != expected_shape:
                        raise FileFormatError("Wrong header: shape of field %s does not match. Got %s, expected %s" % (name, shape, expected_shape))
                    attr.load(f, skip)
                elif words[2].startswith("value="):
                    if not isinstance(attr, ScalarAttr):
                        raise FileFormatError("field '%s' is not a single value." % name)
                    if not skip:
                        if kind == 'i':
                            attr.set(int(words[2][6:]))
                        else:
                            attr.set(float(words[2][6:]))
                else:
                    raise FileFormatError("Malformatted array header line. (shape/value)")

                num_names += 1

            if num_names != len(self._fields) and subset is None:
                raise FileFormatError("Some fields are missing in the file.")