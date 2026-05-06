def _parse_variables_from_code(self):
        """ Parse uniforms, attributes and varyings from the source code.
        """
        
        # Get one string of code with comments removed
        code = '\n\n'.join(self._shaders)
        code = re.sub(r'(.*)(//.*)', r'\1', code, re.M)
        
        # Regexp to look for variable names
        var_regexp = ("\s*VARIABLE\s+"  # kind of variable
                      "((highp|mediump|lowp)\s+)?"  # Precision (optional)
                      "(?P<type>\w+)\s+"  # type
                      "(?P<name>\w+)\s*"  # name
                      "(\[(?P<size>\d+)\])?"  # size (optional)
                      "(\s*\=\s*[0-9.]+)?"  # default value (optional)
                      "\s*;"  # end
                      )
        
        # Parse uniforms, attributes and varyings
        self._code_variables = {}
        for kind in ('uniform', 'attribute', 'varying', 'const'):
            regex = re.compile(var_regexp.replace('VARIABLE', kind),
                               flags=re.MULTILINE)
            for m in re.finditer(regex, code):
                gtype = m.group('type')
                size = int(m.group('size')) if m.group('size') else -1
                this_kind = kind
                if size >= 1:
                    # uniform arrays get added both as individuals and full
                    for i in range(size):
                        name = '%s[%d]' % (m.group('name'), i)
                        self._code_variables[name] = kind, gtype, name, -1
                    this_kind = 'uniform_array'
                name = m.group('name')
                self._code_variables[name] = this_kind, gtype, name, size

        # Now that our code variables are up-to date, we can process
        # the variables that were set but yet unknown.
        self._process_pending_variables()