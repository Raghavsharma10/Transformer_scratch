def bam2es(
        bam_fn,
        es_fo,
        allowed_delta,
    ):
        """Convert BAM file to ES file.

		Args:
			bam_fn (str): File name of the BAM file.
			bam_fo (file): File object of the ES file.
			allowed_delta (int): Maximal allowed coordinates difference for correct reads.
		"""

        es_fo.write("# RN:   read name" + os.linesep)
        es_fo.write("# Q:    is mapped with quality" + os.linesep)
        es_fo.write("# Chr:  chr id" + os.linesep)
        es_fo.write("# D:    direction" + os.linesep)
        es_fo.write("# L:    leftmost nucleotide" + os.linesep)
        es_fo.write("# R:    rightmost nucleotide" + os.linesep)
        es_fo.write("# Cat:  category of alignment assigned by LAVEnder" + os.linesep)
        es_fo.write("#         M_i    i-th segment is correctly mapped" + os.linesep)
        es_fo.write("#         m      segment should be unmapped but it is mapped" + os.linesep)
        es_fo.write("#         w      segment is mapped to a wrong location" + os.linesep)
        es_fo.write("#         U      segment is unmapped and should be unmapped" + os.linesep)
        es_fo.write("#         u      segment is unmapped and should be mapped" + os.linesep)
        es_fo.write("# Segs: number of segments" + os.linesep)
        es_fo.write("# " + os.linesep)
        es_fo.write("# RN\tQ\tChr\tD\tL\tR\tCat\tSegs" + os.linesep)

        with pysam.AlignmentFile(bam_fn, "rb") as sam:
            references_dict = {}

            for i in range(len(sam.references)):
                references_dict[sam.references[i]] = i + 1

            for read in sam:
                rnf_read_tuple = rnftools.rnfformat.ReadTuple()
                rnf_read_tuple.destringize(read.query_name)

                left = read.reference_start + 1
                right = read.reference_end
                chrom_id = references_dict[sam.references[read.reference_id]]

                nb_of_segments = len(rnf_read_tuple.segments)

                if rnf_read_tuple.segments[0].genome_id == 1:
                    should_be_mapped = True
                else:
                    should_be_mapped = False

                # read is unmapped
                if read.is_unmapped:
                    # read should be mapped
                    if should_be_mapped:
                        category = "u"
                    # read should be unmapped
                    else:
                        category = "U"
                # read is mapped
                else:
                    # read should be mapped
                    if should_be_mapped:
                        exists_corresponding_segment = False

                        for j in range(len(rnf_read_tuple.segments)):
                            segment = rnf_read_tuple.segments[j]
                            if (
                                (segment.left == 0 or abs(segment.left - left) <= allowed_delta)
                                and (segment.right == 0 or abs(segment.right - right) <= allowed_delta)
                                and (segment.left != 0 or segment.right == 0)
                                and (chrom_id == 0 or chrom_id == segment.chr_id)
                            ):
                                exists_corresponding_segment = True
                                segment = str(j + 1)
                                break

                        # read was mapped to correct location
                        if exists_corresponding_segment:  # exists ok location?
                            category = "M_" + segment
                        # read was mapped to incorrect location
                        else:
                            category = "w"
                    # read should be unmapped
                    else:
                        category = "m"

                es_fo.write(
                    "\t".join(
                        map(
                            str,
                            [
                                # read name
                                read.query_name,
                                # aligned?
                                "unmapped" if read.is_unmapped else "mapped_" + str(read.mapping_quality),
                                # reference id
                                chrom_id,
                                # direction
                                "R" if read.is_reverse else "F",
                                # left
                                left,
                                # right
                                right,
                                # assigned category
                                category,
                                # count of segments
                                nb_of_segments
                            ]
                        )
                    ) + os.linesep
                )