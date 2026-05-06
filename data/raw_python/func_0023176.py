def _get_replaced_code(self, names):
        """ Return code, with new name, expressions, and replacements applied.
        """
        code = self._code
        
        # Modify name
        fname = names[self]
        code = code.replace(" " + self.name + "(", " " + fname + "(")

        # Apply string replacements first -- these may contain $placeholders
        for key, val in self._replacements.items():
            code = code.replace(key, val)
        
        # Apply assignments to the end of the function
        
        # Collect post lines
        post_lines = []
        for key, val in self._assignments.items():
            if isinstance(key, Variable):
                key = names[key]
            if isinstance(val, ShaderObject):
                val = val.expression(names)
            line = '    %s = %s;' % (key, val)
            post_lines.append(line)
            
        # Add a default $post placeholder if needed
        if 'post' in self._expressions:
            post_lines.append('    $post')
            
        # Apply placeholders for hooks
        post_text = '\n'.join(post_lines)
        if post_text:
            post_text = '\n' + post_text + '\n'
        code = code.rpartition('}')
        code = code[0] + post_text + code[1] + code[2]

        # Add a default $pre placeholder if needed
        if 'pre' in self._expressions:
            m = re.search(fname + r'\s*\([^{]*\)\s*{', code)
            if m is None:
                raise RuntimeError("Cound not find beginning of function '%s'" 
                                   % fname) 
            ind = m.span()[1]
            code = code[:ind] + "\n    $pre\n" + code[ind:]
        
        # Apply template variables
        for key, val in self._expressions.items():
            val = val.expression(names)
            search = r'\$' + key + r'($|[^a-zA-Z0-9_])'
            code = re.sub(search, val+r'\1', code)

        # Done
        if '$' in code:
            v = parsing.find_template_variables(code)
            logger.warning('Unsubstituted placeholders in code: %s\n'
                           '  replacements made: %s', 
                           v, list(self._expressions.keys()))
        
        return code + '\n'