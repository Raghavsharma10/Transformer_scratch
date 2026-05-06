def _apply_identical_sequences_prefilter(self,
                                             seq_path):
        """
            Input : seq_path, a filepath to input FASTA reads
            Method: prepares and writes de-replicated reads
                    to a temporary FASTA file, calls
                    parent method to do the actual
                    de-replication
            Return: exact_match_id_map, a dictionary storing
                    de-replicated amplicon ID as key and
                    all original FASTA IDs with identical
                    sequences as values;
                    unique_seqs_fp, filepath to FASTA file
                    holding only de-replicated sequences
        """
        # creating mapping for de-replicated reads
        seqs_to_cluster, exact_match_id_map =\
            self._prefilter_exact_matches(parse_fasta(seq_path))

        # create temporary file for storing the de-replicated reads
        fd, unique_seqs_fp = mkstemp(
            prefix='SwarmExactMatchFilter', suffix='.fasta')
        close(fd)

        self.files_to_remove.append(unique_seqs_fp)

        # write de-replicated reads to file
        unique_seqs_f = open(unique_seqs_fp, 'w')
        for seq_id, seq in seqs_to_cluster:
            unique_seqs_f.write('>%s_%d\n%s\n'
                                % (seq_id,
                                   len(exact_match_id_map[seq_id]),
                                   seq))
        unique_seqs_f.close()

        return exact_match_id_map, unique_seqs_fp