def gen_rnd_prod_CDR3(self, conserved_J_residues = 'FVW'):
        """Generate a productive CDR3 seq from a Monte Carlo draw of the model.

        Parameters
        ----------
        conserved_J_residues : str, optional
            Conserved amino acid residues defining the CDR3 on the J side (normally
            F, V, and/or W)

        Returns
        -------
        ntseq : str
            Productive CDR3 nucleotide sequence
        aaseq : str
            CDR3 amino acid sequence (aaseq = nt2aa(ntseq))
        V_choice : int
            Index of V allele chosen to generate the CDR3 seq
        J_choice : int
            Index of J allele chosen to generate the CDR3 seq

        """

        coding_pass = False

        while ~coding_pass:
            recomb_events = self.choose_random_recomb_events()
            V_seq = self.cutV_genomic_CDR3_segs[recomb_events['V']]

            #This both checks that the position of the conserved C is
            #identified and that the V isn't fully deleted out of the CDR3
            #region
            if len(V_seq) <= max(recomb_events['delV'], 0):
                continue
            J_seq = self.cutJ_genomic_CDR3_segs[recomb_events['J']]

            #We check that J isn't deleted more than allowed. Note the
            #generative model really should reflect this structure already
            if len(J_seq) < recomb_events['delJ']:
                continue

            V_seq = V_seq[:len(V_seq) - recomb_events['delV']]
            J_seq = J_seq[recomb_events['delJ']:]

            if (len(V_seq)+len(J_seq) + recomb_events['insVJ']) % 3 != 0:
                continue


            insVJ_seq = rnd_ins_seq(recomb_events['insVJ'], self.C_Rvj, self.C_first_nt_bias_insVJ)

            #Translate to amino acid sequence, see if productive
            ntseq = V_seq + insVJ_seq + J_seq
            aaseq = nt2aa(ntseq)

            if '*' not in aaseq and aaseq[0]=='C' and aaseq[-1] in conserved_J_residues:
                return ntseq, aaseq, recomb_events['V'], recomb_events['J']