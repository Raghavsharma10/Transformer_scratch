def convert(profiling_data, outputfile):
    """convert `profiling_data` to calltree format and dump it to `outputfile`

    `profiling_data` can either be:
        - a pstats.Stats instance
        - the filename of a pstats.Stats dump
        - the result of a call to cProfile.Profile.getstats()

    `outputfile` can either be:
        - a file() instance open in write mode
        - a filename
    """
    converter = CalltreeConverter(profiling_data)
    if is_basestring(outputfile):
        with open(outputfile, "w") as f:
            converter.output(f)
    else:
        converter.output(outputfile)