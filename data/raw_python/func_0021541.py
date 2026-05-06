def read_VDJ_palindrome_parameters(self, params_file_name):
        """Read V, D, and J palindrome parameters from file.
        
        Sets the attributes max_delV_palindrome, max_delDl_palindrome,
        max_delDr_palindrome, and max_delJ_palindrome.
    
        Parameters
        ----------
        params_file_name : str
            File name for an IGoR parameter file of a VDJ generative model.
        
        """
        
        params_file = open(params_file_name, 'r')
        
        
        in_delV = False
        in_delDl = False
        in_delDr = False
        in_delJ = False
        
        
        for line in params_file:
            if line.startswith('#Deletion;V_gene;'):
                in_delV = True
                in_delDl = False
                in_delDr = False
                in_delJ = False
            elif line.startswith('#Deletion;D_gene;Three_prime;'):
                in_delV = False
                in_delDl = False
                in_delDr = True
                in_delJ = False
            elif line.startswith('#Deletion;D_gene;Five_prime;'):
                in_delV = False
                in_delDl = True
                in_delDr = False
                in_delJ = False
            elif line.startswith('#Deletion;J_gene;'):
                in_delV = False
                in_delDl = False
                in_delDr = False
                in_delJ = True
            elif any([in_delV, in_delDl, in_delDr, in_delJ]) and line.startswith('%'):
                if int(line.split(';')[-1]) == 0:
                    if in_delV:
                        self.max_delV_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
                    elif in_delDl:
                        self.max_delDl_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
                    elif in_delDr:
                        self.max_delDr_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
                    elif in_delJ:
                        self.max_delJ_palindrome = np.abs(int(line.lstrip('%').split(';')[0]))
            else:
                in_delV = False
                in_delDl = False
                in_delDr = False
                in_delJ = False