def main():
    """pyprf_sim entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'pyprf_sim ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    objNspc = get_arg_parse()

    # Print info if no config argument is provided.
    if any(item is None for item in [objNspc.strCsvPrf, objNspc.strStmApr]):
        print('Please provide necessary file paths, e.g.:')
        print('   pyprf_sim -strCsvPrf /path/to/my_config_file.csv')
        print('             -strStmApr /path/to/my_stim_apertures.npy')

    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        # Call to main function, to invoke pRF analysis:
        pyprf_sim(objNspc.strCsvPrf, objNspc.strStmApr, lgcTest=lgcTest,
                  lgcNoise=objNspc.lgcNoise, lgcRtnNrl=objNspc.lgcRtnNrl,
                  lstRat=objNspc.supsur)