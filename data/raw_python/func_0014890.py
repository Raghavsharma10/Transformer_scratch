def put(self, name_to_val, oned_as='row', on_new_output=None):
        """ Loads a dictionary of variable names into the matlab shell.

        oned_as is the same as in scipy.io.matlab.savemat function:
        oned_as : {'column', 'row'}, optional
        If 'column', write 1-D numpy arrays as column vectors.
        If 'row', write 1D numpy arrays as row vectors.
        """
        self._check_open()
        # We can't give stdin to mlabio.savemat because it needs random access :(
        temp = StringIO()
        mlabio.savemat(temp, name_to_val, oned_as=oned_as)
        temp.seek(0)
        temp_str = temp.read()
        temp.close()
        self.process.stdin.write('load stdio;\n')
        self._read_until('ack load stdio\n', on_new_output=on_new_output)
        self.process.stdin.write(temp_str)
        #print 'sent %d kb' % (len(temp_str) / 1024)
        self._read_until('ack load finished\n', on_new_output=on_new_output)
        self._sync_output(on_new_output=on_new_output)