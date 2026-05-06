def compute_regex_CDR3_template_pgen(self, regex_seq, V_usage_mask_in = None, J_usage_mask_in = None, print_warnings = True, raise_overload_warning = True):
        """Compute Pgen for all seqs consistent with regular expression regex_seq.
    
        Computes Pgen for a (limited vocabulary) regular expression of CDR3 
        amino acid sequences, conditioned on the V genes/alleles indicated in 
        V_usage_mask_in and the J genes/alleles in J_usage_mask_in. Please note
        that this function will list out all the sequences that correspond to the 
        regular expression and then calculate the Pgen of each sequence in
        succession. THIS CAN BE SLOW. Consider defining a custom alphabet to 
        represent any undetermined amino acids as this will greatly speed up the 
        computations. For example, if the symbol ^ is defined as [AGR] in a custom 
        alphabet, then instead of running 
        compute_regex_CDR3_template_pgen('CASS[AGR]SARPEQFF', ppp),
        which will compute Pgen for 3 sequences, the single sequence 
        'CASS^SARPEQFF' can be considered. (Examples are TCRB sequences/model)
        
    
        Parameters
        ----------
        regex_seq : str
            The regular expression string that represents the CDR3 sequences to be 
            listed then their Pgens computed and summed.
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.
        raise_overload_warning : bool
            A flag to warn of more than 10000 seqs corresponding to the regex_seq
    
        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence
        
        Examples
        --------
        >>> generation_probability.compute_regex_CDR3_template_pgen('CASS[AGR]SARPEQFF')
        8.1090898050318022e-10
        >>> generation_probability.compute_regex_CDR3_template_pgen('CASSAX{0,5}SARPEQFF')
        6.8468778040965569e-10
            
        """
        
        V_usage_mask, J_usage_mask = self.format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings)
        
        CDR3_seqs = self.list_seqs_from_regex(regex_seq, print_warnings, raise_overload_warning)
        
        pgen = 0
        for  CDR3_seq in CDR3_seqs:
            if len(CDR3_seq) == 0:
                continue
            pgen += self.compute_CDR3_pgen(CDR3_seq, V_usage_mask, J_usage_mask)
    
        return pgen