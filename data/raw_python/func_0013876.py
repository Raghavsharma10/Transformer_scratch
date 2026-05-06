def _get_var_type_code(self, coltype):
        '''Determines the two-character type code for a given variable type

        Parameters
        ----------
        coltype : type or np.dtype
            The type of the variable

        Returns
        -------
        str
            The variable type code for the given type'''

        if type(coltype) is np.dtype:
            var_type = coltype.kind + str(coltype.itemsize)
            return var_type
        else:
            if coltype is np.int64:
                return 'i8'
            elif coltype is np.int32:
                return 'i4'
            elif coltype is np.int16:
                return 'i2'
            elif coltype is np.int8:
                return 'i1'
            elif coltype is np.uint64:
                return 'u8'
            elif coltype is np.uint32:
                return 'u4'
            elif coltype is np.uint16:
                return 'u2'
            elif coltype is np.uint8:
                return 'u1'
            elif coltype is np.float64:
                return 'f8'
            elif coltype is np.float32:
                return 'f4'
            elif issubclass(coltype, basestring):
                return 'S1'
            else:
                raise TypeError('Unknown Variable Type' + str(coltype))