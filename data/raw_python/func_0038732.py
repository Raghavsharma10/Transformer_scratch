def recode_dwgsim_reads(
        dwgsim_prefix,
        fastq_rnf_fo,
        fai_fo,
        genome_id,
        estimate_unknown_values,
        number_of_read_tuples=10**9,
    ):
        """Convert DwgSim FASTQ file to RNF FASTQ file.

		Args:
			dwgsim_prefix (str): DwgSim prefix of the simulation (see its commandline parameters).
			fastq_rnf_fo (file): File object of RNF FASTQ.
			fai_fo (file): File object for FAI file of the reference genome.
			genome_id (int): RNF genome ID to be used.
			estimate_unknown_values (bool): Estimate unknown values (right coordinate of each end).
			number_of_read_tuples (int): Estimate of number of simulated read tuples (to set width).
		"""

        dwgsim_pattern = re.compile(
            '@(.*)_([0-9]+)_([0-9]+)_([01])_([01])_([01])_([01])_([0-9]+):([0-9]+):([0-9]+)_([0-9]+):([0-9]+):([0-9]+)_(([0-9abcdef])+)'
        )

        ###
        # DWGSIM read name format
        #
        # 1)  contig name (chromsome name)
        # 2)  start end 1 (one-based)
        # 3)  start end 2 (one-based)
        # 4)  strand end 1 (0 - forward, 1 - reverse)
        # 5)  strand end 2 (0 - forward, 1 - reverse)
        # 6)  random read end 1 (0 - from the mutated reference, 1 - random)
        # 7)  random read end 2 (0 - from the mutated reference, 1 - random)
        # 8)  number of sequencing errors end 1 (color errors for colorspace)
        # 9)  number of SNPs end 1
        # 10) number of indels end 1
        # 11) number of sequencing errors end 2 (color errors for colorspace)
        # 12) number of SNPs end 2
        # 13) number of indels end 2
        # 14) read number (unique within a given contig/chromosome)
        ###

        fai_index = rnftools.utils.FaIdx(fai_fo=fai_fo)
        read_tuple_id_width = len(format(number_of_read_tuples, 'x'))

        # parsing FQ file
        read_tuple_id = 0
        last_read_tuple_name = None
        old_fq = "{}.bfast.fastq".format(dwgsim_prefix)

        fq_creator = rnftools.rnfformat.FqCreator(
            fastq_fo=fastq_rnf_fo,
            read_tuple_id_width=read_tuple_id_width,
            genome_id_width=2,
            chr_id_width=fai_index.chr_id_width,
            coor_width=fai_index.coor_width,
            info_reads_in_tuple=True,
            info_simulator="dwgsim",
        )

        i = 0
        with open(old_fq, "r+") as f1:
            for line in f1:
                if i % 4 == 0:
                    read_tuple_name = line[1:].strip()
                    if read_tuple_name != last_read_tuple_name:
                        new_tuple = True
                        if last_read_tuple_name is not None:
                            read_tuple_id += 1
                    else:
                        new_tuple = False

                    last_read_tuple_name = read_tuple_name
                    m = dwgsim_pattern.search(line)
                    if m is None:
                        rnftools.utils.error(
                            "Read tuple '{}' was not created by DwgSim.".format(line[1:]),
                            program="RNFtools",
                            subprogram="MIShmash",
                            exception=ValueError,
                        )

                    contig_name = m.group(1)
                    start_1 = int(m.group(2))
                    start_2 = int(m.group(3))
                    direction_1 = "F" if int(m.group(4)) == 0 else "R"
                    direction_2 = "F" if int(m.group(5)) == 0 else "R"
                    # random_1 = bool(m.group(6))
                    # random_2 = bool(m.group(7))
                    # seq_err_1 = int(m.group(8))
                    # snp_1 = int(m.group(9))
                    # indels_1 = int(m.group(10))
                    # seq_err_2 = int(m.group(11))
                    # snp_2 = int(m.group(12))
                    # indels_2 = int(m.group(13))
                    # read_tuple_id_dwg = int(m.group(14), 16)

                    chr_id = fai_index.dict_chr_ids[contig_name] if fai_index.dict_chr_ids != {} else "0"

                elif i % 4 == 1:
                    bases = line.strip()

                    if new_tuple:

                        segment = rnftools.rnfformat.Segment(
                            genome_id=genome_id,
                            chr_id=chr_id,
                            direction=direction_1,
                            left=start_1,
                            right=start_1 + len(bases) - 1 if estimate_unknown_values else 0,
                        )

                    else:

                        segment = rnftools.rnfformat.Segment(
                            genome_id=genome_id,
                            chr_id=chr_id,
                            direction=direction_2,
                            left=start_2,
                            right=start_2 + len(bases) - 1 if estimate_unknown_values else 0,
                        )

                elif i % 4 == 2:
                    pass

                elif i % 4 == 3:
                    qualities = line.strip()
                    fq_creator.add_read(
                        read_tuple_id=read_tuple_id,
                        bases=bases,
                        qualities=qualities,
                        segments=[segment],
                    )

                i += 1

        fq_creator.flush_read_tuple()