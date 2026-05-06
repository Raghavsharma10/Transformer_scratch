def get(self,
                    names_to_get,
                    extract_numpy_scalars=True,
                    on_new_output=None):
        """ Loads the requested variables from the matlab shell.

        names_to_get can be either a variable name, a list of variable names, or
        None.
        If it is a variable name, the values is returned.
        If it is a list, a dictionary of variable_name -> value is returned.
        If it is None, a dictionary with all variables is returned.

        If extract_numpy_scalars is true, the method will convert numpy scalars
        (0-dimension arrays) to a regular python variable.
        """
        self._check_open()
        single_item = isinstance(names_to_get, (unicode, str))
        if single_item:
            names_to_get = [names_to_get]
        if names_to_get == None:
            self.process.stdin.write('save stdio;\n')
        else:
            # Make sure that we throw an excpetion if the names are not defined.
            for name in names_to_get:
                self.eval('%s;' % name, print_expression=False, on_new_output=on_new_output)
            #print 'save(\'stdio\', \'%s\');\n' % '\', \''.join(names_to_get)
            self.process.stdin.write(
                "save('stdio', '%s', '-v7');\n" % '\', \''.join(names_to_get))
        # We have to read to a temp buffer because mlabio.loadmat needs
        # random access :(
        self._read_until('start_binary\n', on_new_output=on_new_output)
        #print 'got start_binary'
        temp_str = self._sync_output(on_new_output=on_new_output)
        #print 'got all outout'
        # Remove expected output and "\n>>"
        # TODO(dani): Get rid of the unecessary copy.
        # MATLAB 2010a adds an extra >> so we need to remove more spaces.
        if self.matlab_version == (2010, 'a'):
            temp_str = temp_str[:-len(self.expected_output_end)-6]
        else:
            temp_str = temp_str[:-len(self.expected_output_end)-3]
        temp = StringIO(temp_str)
        #print ('____')
        #print len(temp_str)
        #print ('____')
        ret = mlabio.loadmat(temp, chars_as_strings=True, squeeze_me=True)
        #print '******'
        #print ret
        #print '******'
        temp.close()
        if single_item:
            return ret.values()[0]
        for key in ret.iterkeys():
            while ret[key].shape and ret[key].shape[-1] == 1:
                ret[key] = ret[key][0]
            if extract_numpy_scalars:
                if isinstance(ret[key], np.ndarray) and not ret[key].shape:
                    ret[key] = ret[key].tolist()
        #print 'done'
        return ret