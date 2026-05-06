def qry_coords_from_ref_coord(self, ref_coord, variant_list):
        '''Given a reference position and a list of variants ([variant.Variant]),
           works out the position in the query sequence, accounting for indels.
           Returns a tuple: (position, True|False), where second element is whether
           or not the ref_coord lies in an indel. If it is, then
           returns the corresponding start position
           of the indel in the query'''
        if self.ref_coords().distance_to_point(ref_coord) > 0:
            raise Error('Cannot get query coord in qry_coords_from_ref_coord because given ref_coord ' + str(ref_coord) + ' does not lie in nucmer alignment:\n' + str(self))

        indel_variant_indexes = []

        for i in range(len(variant_list)):
            if variant_list[i].var_type not in {variant.INS, variant.DEL}:
                continue
            if not self.intersects_variant(variant_list[i]):
                continue
            if variant_list[i].ref_start <= ref_coord <= variant_list[i].ref_end:
                return variant_list[i].qry_start, True
            elif variant_list[i].ref_start < ref_coord:
                indel_variant_indexes.append(i)

        distance = ref_coord - min(self.ref_start, self.ref_end)

        for i in indel_variant_indexes:
            if variant_list[i].var_type == variant.INS:
                distance += len(variant_list[i].qry_base)
            else:
                assert variant_list[i].var_type == variant.DEL
                distance -= len(variant_list[i].ref_base)

        if self.on_same_strand():
            return min(self.qry_start, self.qry_end) + distance, False
        else:
            return max(self.qry_start, self.qry_end) - distance, False