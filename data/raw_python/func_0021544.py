def load_and_process_igor_model(self, marginals_file_name):
        """Set attributes by reading a generative model from IGoR marginal file.
        
        Sets attributes PVJ, PdelV_given_V, PdelJ_given_J, PinsVJ, and Rvj.
        
        Parameters
        ----------
        marginals_file_name : str
            File name for a IGoR model marginals file.
        
        """
        
        raw_model = read_igor_marginals_txt(marginals_file_name)
        
        self.PinsVJ = raw_model[0]['vj_ins']
        self.PdelV_given_V = raw_model[0]['v_3_del'].T
        self.PdelJ_given_J = raw_model[0]['j_5_del'].T
        self.PVJ = np.multiply( raw_model[0]['j_choice'].T, raw_model[0]['v_choice']).T
        Rvj_raw = raw_model[0]['vj_dinucl'].reshape((4, 4)).T
        self.Rvj = np.multiply(Rvj_raw, 1/np.sum(Rvj_raw, axis = 0))