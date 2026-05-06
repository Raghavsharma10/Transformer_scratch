def parse_nem_rows(nem_list: Iterable, file_name=None) -> NEMFile:
    """ Parse NEM row iterator and return meter readings named tuple """

    header = HeaderRecord(None, None, None, None, file_name)
    readings = dict()  # readings nested by NMI then channel
    trans = dict()  # transactions nested by NMI then channel
    nmi_d = None  # current NMI details block that readings apply to

    for i, row in enumerate(nem_list):
        record_indicator = int(row[0])

        if i == 0 and record_indicator != 100:
            raise ValueError("NEM Files must start with a 100 row")

        if record_indicator == 100:
            header = parse_100_row(row, file_name)
            if header.version_header not in ['NEM12', 'NEM13']:
                raise ValueError("Invalid NEM version {}".format(
                    header.version_header))

        elif record_indicator == 900:
            for nmi in readings:
                for suffix in readings[nmi]:
                    readings[nmi][suffix] = flatten_list(readings[nmi][suffix])
            break  # End of file

        elif header.version_header == 'NEM12' and record_indicator == 200:
            try:
                nmi_details = parse_200_row(row)
            except ValueError:
                logging.error('Error passing 200 row:')
                logging.error(row)
                raise
            nmi_d = nmi_details

            if nmi_d.nmi not in readings:
                readings[nmi_d.nmi] = {}
            if nmi_d.nmi_suffix not in readings[nmi_d.nmi]:
                readings[nmi_d.nmi][nmi_d.nmi_suffix] = []
            if nmi_d.nmi not in trans:
                trans[nmi_d.nmi] = {}
            if nmi_d.nmi_suffix not in trans[nmi_d.nmi]:
                trans[nmi_d.nmi][nmi_d.nmi_suffix] = []

        elif header.version_header == 'NEM12' and record_indicator == 300:
            num_intervals = int(24 * 60 / nmi_d.interval_length)
            assert len(row) > num_intervals, "Incomplete 300 Row in {}".format(
                file_name)
            interval_record = parse_300_row(row, nmi_d.interval_length,
                                            nmi_d.uom)
            # don't flatten the list of interval readings at this stage,
            # as they may need to be adjusted by a 400 row
            readings[nmi_d.nmi][nmi_d.nmi_suffix].append(
                interval_record.interval_values)

        elif header.version_header == 'NEM12' and record_indicator == 400:
            event_record = parse_400_row(row)
            readings[nmi_d.nmi][nmi_d.nmi_suffix][-1] = update_reading_events(
                readings[nmi_d.nmi][nmi_d.nmi_suffix][-1], event_record)

        elif header.version_header == 'NEM12' and record_indicator == 500:
            b2b_details = parse_500_row(row)
            trans[nmi_d.nmi][nmi_d.nmi_suffix].append(b2b_details)

        elif header.version_header == 'NEM13' and record_indicator == 550:
            b2b_details = parse_550_row(row)
            trans[nmi_d.nmi][nmi_d.nmi_suffix].append(b2b_details)

        elif header.version_header == 'NEM13' and record_indicator == 250:
            basic_data = parse_250_row(row)
            reading = calculate_manual_reading(basic_data)

            nmi_d = basic_data

            if basic_data.nmi not in readings:
                readings[nmi_d.nmi] = {}
            if nmi_d.nmi_suffix not in readings[nmi_d.nmi]:
                readings[nmi_d.nmi][nmi_d.nmi_suffix] = []
            if nmi_d.nmi not in trans:
                trans[nmi_d.nmi] = {}
            if nmi_d.nmi_suffix not in trans[nmi_d.nmi]:
                trans[nmi_d.nmi][nmi_d.nmi_suffix] = []

            readings[nmi_d.nmi][nmi_d.nmi_suffix].append([reading])

        else:
            logging.warning(
                "Record indicator %s not supported and was skipped",
                record_indicator)
    return NEMFile(header, readings, trans)