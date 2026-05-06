def recode_sam_reads(
        sam_fn,
        fastq_rnf_fo,
        fai_fo,
        genome_id,
        number_of_read_tuples=10**9,
        simulator_name=None,
        allow_unmapped=False,
    ):
        """Transform a SAM file to RNF-compatible FASTQ.

		Args:
			sam_fn (str): SAM/BAM file - file name.
			fastq_rnf_fo (str): Output FASTQ file - file object.
			fai_fo (str): FAI index of the reference genome - file object.
			genome_id (int): Genome ID for RNF.
			number_of_read_tuples (int): Expected number of read tuples (to set width of read tuple id).
			simulator_name (str): Name of the simulator. Used for comment in read tuple name.
			allow_unmapped (bool): Allow unmapped reads.

		Raises:
			NotImplementedError
		"""

        fai_index = rnftools.utils.FaIdx(fai_fo)
        # last_read_tuple_name=[]
        read_tuple_id_width = len(format(number_of_read_tuples, 'x'))
        fq_creator = rnftools.rnfformat.FqCreator(
            fastq_fo=fastq_rnf_fo,
            read_tuple_id_width=read_tuple_id_width,
            genome_id_width=2,
            chr_id_width=fai_index.chr_id_width,
            coor_width=fai_index.coor_width,
            info_reads_in_tuple=True,
            info_simulator=simulator_name,
        )

        # todo: check if clipping corrections is well implemented
        cigar_reg_shift = re.compile("([0-9]+)([MDNP=X])")

        # todo: other upac codes
        reverse_complement_dict = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }

        read_tuple_id = 0
        last_read_tuple_name = None
        with pysam.AlignmentFile(
            sam_fn,
            check_header=False,
        ) as samfile:
            for alignment in samfile:
                if alignment.query_name != last_read_tuple_name and last_read_tuple_name is not None:
                    read_tuple_id += 1
                last_read_tuple_name = alignment.query_name

                if alignment.is_unmapped:
                    rnftools.utils.error(
                        "SAM files used for conversion should not contain unaligned segments. "
                        "This condition is broken by read tuple "
                        "'{}' in file '{}'.".format(alignment.query_name, sam_fn),
                        program="RNFtools",
                        subprogram="MIShmash",
                        exception=NotImplementedError,
                    )

                if alignment.is_reverse:
                    direction = "R"
                    bases = "".join([reverse_complement_dict[nucl] for nucl in alignment.seq[::-1]])
                    qualities = str(alignment.qual[::-1])
                else:
                    direction = "F"
                    bases = alignment.seq[:]
                    qualities = str(alignment.qual[:])

                # todo: are chromosomes in bam sorted correctly (the same order as in FASTA)?
                if fai_index.dict_chr_ids != {}:
                    chr_id = fai_index.dict_chr_ids[samfile.getrname(alignment.reference_id)]
                else:
                    chr_id = "0"

                left = int(alignment.reference_start) + 1
                right = left - 1
                for (steps, operation) in cigar_reg_shift.findall(alignment.cigarstring):
                    right += int(steps)

                segment = rnftools.rnfformat.Segment(
                    genome_id=genome_id,
                    chr_id=chr_id,
                    direction=direction,
                    left=left,
                    right=right,
                )

                fq_creator.add_read(
                    read_tuple_id=read_tuple_id,
                    bases=bases,
                    qualities=qualities,
                    segments=[segment],
                )
        fq_creator.flush_read_tuple()