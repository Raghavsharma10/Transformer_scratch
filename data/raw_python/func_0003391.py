def loadfromfile(self, filelike):
        """
        Read configurations from a file-like object, or a sequence of strings. Old values are not
        cleared, if you want to reload the configurations completely, you should call `clear()`
        before using `load*` methods.
        """
        line_format = re.compile(r'((?:[a-zA-Z][a-zA-Z0-9_]*\.)*[a-zA-Z][a-zA-Z0-9_]*)\s*=\s*')
        space = re.compile(r'\s')
        line_no = 0
        # Suport multi-line data
        line_buffer = []
        line_key = None
        last_line_no = None
        for l in filelike:
            line_no += 1
            ls = l.strip()
            # If there is a # in start, the whole line is remarked.
            # ast.literal_eval already processed any # after '='.
            if not ls or ls.startswith('#'):
                continue
            if space.match(l):
                if not line_key:
                    # First line cannot start with space
                    raise ValueError('Error format in line %d: first line cannot start with space%s\n' % (line_no, l))
                line_buffer.append(l)
            else:
                if line_key:
                    # Process buffered lines
                    try:
                        value = ast.literal_eval(''.join(line_buffer))
                    except Exception:
                        typ, val, tb = sys.exc_info()
                        raise ValueError('Error format in line %d(%s: %s):\n%s' % (last_line_no, typ.__name__, str(val), ''.join(line_buffer)))
                    self[line_key] = value
                # First line
                m = line_format.match(l)
                if not m:
                    raise ValueError('Error format in line %d:\n%s' % (line_no, l))
                line_key = m.group(1)
                last_line_no = line_no
                del line_buffer[:]
                line_buffer.append(l[m.end():])
        if line_key:
            # Process buffered lines
            try:
                value = ast.literal_eval(''.join(line_buffer))
            except Exception:
                typ, val, tb = sys.exc_info()
                raise ValueError('Error format in line %d(%s: %s):\n%s' % (last_line_no, typ.__name__, str(val), ''.join(line_buffer)))
            self[line_key] = value