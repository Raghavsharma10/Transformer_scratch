def et2roc(et_fo, roc_fo):
        """ET to ROC conversion.

		Args:
			et_fo (file): File object for the ET file.
			roc_fo (file): File object for the ROC file.

		raises: ValueError

		"""

        stats_dicts = [
            {
                "q": q,
                "M": 0,
                "w": 0,
                "m": 0,
                "P": 0,
                "U": 0,
                "u": 0,
                "T": 0,
                "t": 0,
                "x": 0
            } for q in range(rnftools.lavender.MAXIMAL_MAPPING_QUALITY + 1)
        ]

        for line in et_fo:
            line = line.strip()
            if line != "" and line[0] != "#":
                (read_tuple_name, tab, info_categories) = line.partition("\t")
                intervals = info_categories.split(",")
                for interval in intervals:
                    category = interval[0]
                    (left, colon, right) = interval[2:].partition("-")
                    for q in range(int(left), int(right) + 1):
                        stats_dicts[q][category] += 1

        roc_fo.write("# Numbers of reads in several categories in dependence" + os.linesep)
        roc_fo.write("# on the applied threshold on mapping quality q" + os.linesep)
        roc_fo.write("# " + os.linesep)
        roc_fo.write("# Categories:" + os.linesep)
        roc_fo.write("#        M: Mapped correctly." + os.linesep)
        roc_fo.write("#        w: Mapped to a wrong position." + os.linesep)
        roc_fo.write("#        m: Mapped but should be unmapped." + os.linesep)
        roc_fo.write("#        P: Multimapped." + os.linesep)
        roc_fo.write("#        U: Unmapped and should be unmapped." + os.linesep)
        roc_fo.write("#        u: Unmapped but should be mapped." + os.linesep)
        roc_fo.write("#        T: Thresholded correctly." + os.linesep)
        roc_fo.write("#        t: Thresholded incorrectly." + os.linesep)
        roc_fo.write("#        x: Unknown." + os.linesep)
        roc_fo.write("#" + os.linesep)
        roc_fo.write("# q\tM\tw\tm\tP\tU\tu\tT\tt\tx\tall" + os.linesep)

        l_numbers = []
        for line in stats_dicts:
            numbers = [
                line["M"], line["w"], line["m"], line["P"], line["U"], line["u"], line["T"], line["t"], line["x"]
            ]
            if numbers != l_numbers:
                roc_fo.write("\t".join([str(line["q"])] + list(map(str, numbers)) + [str(sum(numbers))]) + os.linesep)
            l_numbers = numbers