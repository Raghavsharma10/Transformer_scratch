def read_igor_VJ_palindrome_parameters(self, params_file_name):
        """Read V and J palindrome parameters from file.
        
        Sets the attributes max_delV_palindrome and max_delJ_palindrome.
    
        Parameters
        ----------
        params_file_name : str
            File name for an IGoR parameter file of a VJ generative model.
        
        """
        params_file = open(params_file_name, 'r')
        
        
        in_delV = False
        in_delJ = False
        
        
        for line in params_file:
            if line.startswith('#Deletion;V_gene;'):
                in_delV = True
                in_delJ = False
            elif line.startswith('#Deletion;J_gene;'):
                in_delV = False
                in_delJ = True
            elif any([in_delV, in_delJ]) and line.startswith('%'):
                if int(line.split(';')[-1]) == 0:
                    if in_delV:
                        self.max_delV_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
                    elif in_delJ:
                        self.max_delJ_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
            else:
                in_delV = False
                in_delJ = False