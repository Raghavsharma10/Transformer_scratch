def _convertBlastParamsToDict(self, record):
        """
        Pull the global BLAST parameters out of a BLAST record and return
        them as a C{dict}.

        Some of these attributes are useless (not filled in), but we record
        them all just in case we one day need them or they start to be used or
        they disappear etc. Any of those changes might alert us that something
        has changed in BLAST XML output or in BioPython.

        @param record: An instance of C{Bio.Blast.Record.Blast}. The attributes
            on this don't seem to be documented. You'll need to look at the
            BioPython source to see everything it contains.
        @return: A C{dict}, as described above.
        """
        result = {}
        for attr in (
                # From Bio.Blast.Record.Header
                'application',
                'version',
                'date',
                'reference',
                'query',
                'query_letters',
                'database',
                'database_sequences',
                'database_letters',
                # From Bio.Blast.Record.DatabaseReport
                'database_name',
                'posted_date',
                'num_letters_in_database',
                'num_sequences_in_database',
                'ka_params',
                'gapped',
                'ka_params_gap',
                # From Bio.Blast.Record.Parameters
                'matrix',
                'gap_penalties',
                'sc_match',
                'sc_mismatch',
                'num_hits',
                'num_sequences',
                'num_good_extends',
                'num_seqs_better_e',
                'hsps_no_gap',
                'hsps_prelim_gapped',
                'hsps_prelim_gapped_attemped',
                'hsps_gapped',
                'query_id',
                'query_length',
                'database_length',
                'effective_hsp_length',
                'effective_query_length',
                'effective_database_length',
                'effective_search_space',
                'effective_search_space_used',
                'frameshift',
                'threshold',
                'window_size',
                'dropoff_1st_pass',
                'gap_x_dropoff',
                'gap_x_dropoff_final',
                'gap_trigger',
                'blast_cutoff'):
            result[attr] = getattr(record, attr)
        return result