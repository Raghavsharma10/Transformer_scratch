def to_msp_crunch(self):
        '''Returns the alignment as a line in MSPcrunch format. The columns are space-separated and are:
           1. score
           2. percent identity
           3. match start in the query sequence
           4. match end in the query sequence
           5. query sequence name
           6. subject sequence start
           7. subject sequence end
           8. subject sequence name'''

        # we don't know the alignment score. Estimate it. This approximates 1 for a match.
        aln_score = int(self.percent_identity * 0.005 * (self.hit_length_ref + self.hit_length_qry))

        return ' '.join(str(x) for x in [
                aln_score,
                '{0:.2f}'.format(self.percent_identity),
                self.qry_start + 1,
                self.qry_end + 1,
                self.qry_name,
                self.ref_start + 1,
                self.ref_end + 1,
                self.ref_name
        ])