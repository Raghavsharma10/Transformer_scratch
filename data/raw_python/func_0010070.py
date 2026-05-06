def write_line(self, d, fmt=SPARSE):
        """
        Converts a single data line to a string.
        """
        
        def smart_quote(s):
            if isinstance(s, basestring) and ' ' in s and s[0] != '"':
                s = '"%s"' % s
            return s
        
        if fmt == DENSE:
            #TODO:fix
            assert not isinstance(d, dict), NotImplemented
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at in NUMERIC_TYPES:
                    line.append(str(e))
                elif at == TYPE_STRING:
                    line.append(self.esc(e))
                elif at == TYPE_NOMINAL:
                    line.append(e)
                else:
                    raise Exception("Type " + at + " not supported for writing!")
            s = ','.join(map(str, line))
            return s
        elif fmt == SPARSE:
            line = []
            
            # Convert flat row into dictionary.
            if isinstance(d, (list, tuple)):
                d = dict(zip(self.attributes, d))
                for k in d:
                    at = self.attribute_types.get(k)
                    if isinstance(d[k], Value):
                        continue
                    elif d[k] == MISSING:
                        d[k] = Str(d[k])
                    elif at in (TYPE_NUMERIC, TYPE_REAL):
                        d[k] = Num(d[k])
                    elif at == TYPE_STRING:
                        d[k] = Str(d[k])
                    elif at == TYPE_INTEGER:
                        d[k] = Int(d[k])
                    elif at == TYPE_NOMINAL:
                        d[k] = Nom(d[k])
                    elif at == TYPE_DATE:
                        d[k] = Date(d[k])
                    else:
                        raise Exception('Unknown type: %s' % at)

            for i, name in enumerate(self.attributes):
                v = d.get(name)
                if v is None:
#                    print 'Skipping attribute with None value:', name
                    continue
                elif v == MISSING or (isinstance(v, Value) and v.value == MISSING):
                    v = MISSING
                elif isinstance(v, String):
                    v = '"%s"' % v.value
                elif isinstance(v, Date):
                    date_format = self.attribute_data.get(name, DEFAULT_DATE_FORMAT)
                    date_format = convert_weka_to_py_date_pattern(date_format)
                    if isinstance(v.value, basestring):
                        _value = dateutil.parser.parse(v.value)
                    else:
                        assert isinstance(v.value, (date, datetime))
                        _value = v.value
                    v.value = v = _value.strftime(date_format)
                elif isinstance(v, Value):
                    v = v.value

                if v != MISSING and self.attribute_types[name] == TYPE_NOMINAL and str(v) not in map(str, self.attribute_data[name]):
                    pass
                else:
                    line.append('%i %s' % (i, smart_quote(v)))

            if len(line) == 1 and MISSING in line[-1]:
                # Skip lines with nothing other than a missing class.
                return
            elif not line:
                # Don't write blank lines.
                return
            return '{' + (', '.join(line)) + '}'
        else:
            raise Exception('Uknown format: %s' % (fmt,))