def process_summary(summaryfile, **kwargs):
    """Extracting information from an albacore summary file.

    Only reads which have a >0 length are returned.

    The fields below may or may not exist, depending on the type of sequencing performed.
    Fields 1-14 are for 1D sequencing.
    Fields 1-23 for 2D sequencing.
    Fields 24-27, 2-5, 22-23 for 1D^2 (1D2) sequencing
    Fields 28-38 for barcoded workflows
     1  filename
     2  read_id
     3  run_id
     4  channel
     5  start_time
     6  duration
     7  num_events
     8  template_start
     9  num_events_template
    10  template_duration
    11  num_called_template
    12  sequence_length_template
    13  mean_qscore_template
    14  strand_score_template
    15  complement_start
    16    num_events_complement
    17    complement_duration
    18    num_called_complement
    19    sequence_length_complement
    20    mean_qscore_complement
    21    strand_score_complement
    22    sequence_length_2d
    23    mean_qscore_2d
    24    filename1
    25    filename2
    26    read_id1
    27    read_id2
    28    barcode_arrangement
    29    barcode_score
    30    barcode_full_arrangement
    31    front_score
    32    rear_score
    33    front_begin_index
    34    front_foundseq_length
    35    rear_end_index
    36    rear_foundseq_length
    37    kit
    38    variant
    """
    logging.info("Nanoget: Collecting metrics from summary file {} for {} sequencing".format(
        summaryfile, kwargs["readtype"]))
    ut.check_existance(summaryfile)
    if kwargs["readtype"] == "1D":
        cols = ["read_id", "run_id", "channel", "start_time", "duration",
                "sequence_length_template", "mean_qscore_template"]
    elif kwargs["readtype"] in ["2D", "1D2"]:
        cols = ["read_id", "run_id", "channel", "start_time", "duration",
                "sequence_length_2d", "mean_qscore_2d"]
    if kwargs["barcoded"]:
        cols.append("barcode_arrangement")
        logging.info("Nanoget: Extracting metrics per barcode.")
    try:
        datadf = pd.read_csv(
            filepath_or_buffer=summaryfile,
            sep="\t",
            usecols=cols,
        )
    except ValueError:
        logging.error("Nanoget: did not find expected columns in summary file {}:\n {}".format(
            summaryfile, ', '.join(cols)))
        sys.exit("ERROR: expected columns in summary file {} not found:\n {}".format(
            summaryfile, ', '.join(cols)))
    if kwargs["barcoded"]:
        datadf.columns = ["readIDs", "runIDs", "channelIDs", "time", "duration",
                          "lengths", "quals", "barcode"]
    else:
        datadf.columns = ["readIDs", "runIDs", "channelIDs", "time", "duration", "lengths", "quals"]
    logging.info("Nanoget: Finished collecting statistics from summary file {}".format(summaryfile))
    return ut.reduce_memory_usage(datadf.loc[datadf["lengths"] != 0].copy())