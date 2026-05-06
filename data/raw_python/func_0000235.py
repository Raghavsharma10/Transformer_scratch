def _analyse_mat_sections(sections):
        """
        Cases:
        - ICRU flag present, LOADDEDX flag missing -> data loaded from some data hardcoded in SH12A binary,
        no need to load external files
        - ICRU flag present, LOADDEDX flag present -> data loaded from external files. ICRU number read from ICRU flag,
        any number following LOADDEDX flag is ignored.
        - ICRU flag missing, LOADDEDX flag present -> data loaded from external files. ICRU number read from LOADDEDX
        - ICRU flag missing, LOADDEDX flag missing -> nothing happens
        """
        icru_numbers = []
        for section in sections:
            load_present = False
            load_value = False
            icru_value = False
            for e in section:
                split_line = e.split()
                if "LOADDEDX" in e:
                    load_present = True
                    if len(split_line) > 1:
                        load_value = split_line[1] if "!" not in split_line[1] else False  # ignore ! comments
                elif "ICRU" in e and len(split_line) > 1:
                    icru_value = split_line[1] if "!" not in split_line[1] else False  # ignore ! comments
            if load_present:  # LOADDEDX is present, so external file is required
                if icru_value:  # if ICRU value was given
                    icru_numbers.append(icru_value)
                elif load_value:  # if only LOADDEDX with values was present in section
                    icru_numbers.append(load_value)
        return icru_numbers