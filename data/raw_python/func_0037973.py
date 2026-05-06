def es2et(
        es_fo,
        et_fo,
    ):
        """Convert ES to ET.

		Args:
			es_fo (file): File object for the ES file.
			et_fo (file): File object for the ET file.
		"""

        et_fo.write("# Mapping information for read tuples" + os.linesep)
        et_fo.write("#" + os.linesep)
        et_fo.write("# RN:   read name" + os.linesep)
        et_fo.write("# I:    intervals with asigned categories" + os.linesep)
        et_fo.write("#" + os.linesep)
        et_fo.write("# RN	I" + os.linesep)

        last_rname = ""
        for line in es_fo:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            else:
                (rname, mapped, ref, direction, left, right, category, nb_of_segments) = line.split("\t")
                nb_of_segments = int(nb_of_segments)

                # print(rname,last_rname,mapped)
                # new read
                if rname != last_rname:
                    # update
                    if last_rname != "":
                        voc = Bam._vector_of_categories(single_reads_statistics, rname, nb_of_segments)
                        et_fo.write(Bam._et_line(readname=rname, vector_of_categories=voc))
                        et_fo.write(os.linesep)

                    # nulling
                    single_reads_statistics = [
                        {
                            "U": 0,
                            "u": 0,
                            "M": [],
                            "m": 0,
                            "w": 0,
                            "T": 0,
                            "t": 0,
                        } for i in range(rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1)
                    ]
                    last_rname = rname

                ####################
                # Unmapped segment #
                ####################

                #####
                # U #
                #####
                if category == "U":
                    for q in range(len(single_reads_statistics)):
                        single_reads_statistics[q]["U"] += 1

                #####
                # u #
                #####
                elif category == "u":
                    for q in range(len(single_reads_statistics)):
                        single_reads_statistics[q]["u"] += 1

                ##################
                # Mapped segment #
                ##################

                else:
                    mapping_quality = int(mapped.replace("mapped_", ""))
                    assert 0 <= mapping_quality and mapping_quality <= rnftools.lavender.MAXIMAL_MAPPING_QUALITY, mapping_quality

                    #####
                    # m #
                    #####
                    if category == "m":
                        for q in range(mapping_quality + 1):
                            single_reads_statistics[q]["m"] += 1
                        for q in range(mapping_quality + 1, rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1):
                            single_reads_statistics[q]["T"] += 1

                    #####
                    # w #
                    #####
                    elif category == "w":
                        for q in range(mapping_quality + 1):
                            single_reads_statistics[q]["w"] += 1
                        for q in range(mapping_quality + 1, rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1):
                            single_reads_statistics[q]["t"] += 1

                    #####
                    # M #
                    #####
                    else:
                        assert category[0] == "M", category
                        segment_id = int(category.replace("M_", ""))
                        for q in range(mapping_quality + 1):
                            single_reads_statistics[q]["M"].append(segment_id)
                        for q in range(mapping_quality + 1, rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1):
                            single_reads_statistics[q]["t"] += 1

        # last read
        voc = Bam._vector_of_categories(single_reads_statistics, rname, nb_of_segments)
        et_fo.write(Bam._et_line(readname=rname, vector_of_categories=voc))
        et_fo.write(os.linesep)