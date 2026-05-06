def update_indel(self, nucmer_snp):
        '''Indels are reported over multiple lines, 1 base insertion or deletion per line. This method extends the current variant by 1 base if it's an indel and adjacent to the new SNP and returns True. If the current variant is a SNP, does nothing and returns False'''
        new_variant = Variant(nucmer_snp)
        if self.var_type not in [INS, DEL] \
          or self.var_type != new_variant.var_type \
          or self.qry_name != new_variant.qry_name \
          or self.ref_name != new_variant.ref_name \
          or self.reverse != new_variant.reverse:
            return False
        if self.var_type == INS \
          and self.ref_start == new_variant.ref_start \
          and self.qry_end + 1 == new_variant.qry_start:
            self.qry_base += new_variant.qry_base
            self.qry_end += 1
            return True
        if self.var_type == DEL \
          and self.qry_start == new_variant.qry_start \
          and self.ref_end + 1 == new_variant.ref_start:
            self.ref_base += new_variant.ref_base
            self.ref_end += 1
            return True

        return False