def pop(self, name):
        """Remove and return metadata about variable

        Parameters
        ----------
        name : str
            variable name

        Returns
        -------
        pandas.Series
            Series of metadata for variable
        """
        # check if present
        if name in self:
            # get case preserved name for variable
            new_name = self.var_case_name(name)
            # check if 1D or nD
            if new_name in self.keys():
                output = self[new_name]
                self.data.drop(new_name, inplace=True, axis=0)
            else:
                output = self.ho_data.pop(new_name)
                
            return output
        else:
            raise KeyError('Key not present in metadata variables')