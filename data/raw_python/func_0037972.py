def _vector_of_categories(srs, read_tuple_name, parts):
        """Create vector of categories (voc[q] ... assigned category for given quality level)

		srs ... single read statistics ... for every q ... dictionary
		read_tuple_name ... read name
		parts ... number of segments
		"""

        # default value
        vec = ["x" for i in range(rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1)]
        assert len(srs) <= rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1, srs

        should_be_mapped = bool(srs[0]["m"] + srs[0]["U"] == 0)

        for q in range(len(srs)):

            #####
            # M # - all parts correctly aligned
            #####
            if len(srs[q]["M"]
                   ) == parts and srs[q]["w"] == 0 and srs[q]["m"] == 0 and srs[q]["U"] == 0 and srs[q]["u"] == 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "M"

            #####
            # w # - at least one segment is incorrectly aligned
            #####
            if srs[q]["w"] > 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "w"

            #####
            # m # - at least one segment was aligned but should not be aligned
            #####
            if srs[q]["w"] == 0 and srs[q]["m"] > 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "m"

            #####
            # U # - all segments should be unaligned but are unaligned
            #####
            if srs[q]["U"] > 0 and srs[q]["u"] == 0 and srs[q]["m"] == 0 and srs[q]["w"] == 0 and len(srs[q]["M"]) == 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "U"

            #####
            # u # - at least one segment was unaligned but should be aligned
            #####
            if srs[q]["w"] == 0 and srs[q]["u"] > 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "u"

            #####
            # t # - at least one segment was thresholded
            #####
            if len(srs[q]["M"]) != parts and srs[q]["w"] == 0 and srs[q]["m"] == 0 and srs[q]["U"] == 0 and srs[q][
                "u"] == 0 and srs[q]["t"] > 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "t"

            #####
            # T # - at least one segment was thresholded
            #####
            if len(srs[q]["M"]) != parts and srs[q]["w"] == 0 and srs[q]["m"] == 0 and srs[q]["U"] == 0 and srs[q][
                "u"] == 0 and srs[q]["T"] > 0:
                assert vec[q] == "x", str((q, srs[q]))
                vec[q] = "T"

            #####
            # P # - multimapped, M + w + m > parts
            #####

            # only this one can rewrite some older assignment
            if len(srs[q]["M"]) + srs[q]["w"] + srs[q]["m"] > parts and srs[q]["U"] == 0 and srs[q]["u"] == 0:
                # assert vec[q]=="x",str((q,srs[q]))
                vec[q] = "P"

        return vec