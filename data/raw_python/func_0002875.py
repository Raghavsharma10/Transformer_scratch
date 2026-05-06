def get_arg_parse():
    """Parses the Command Line Arguments using argparse."""
    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace -config file path:
    objParser.add_argument('-config', required=True,
                           metavar='/path/to/config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # Add argument to namespace -prior results file path:
    objParser.add_argument('-strPthPrior', required=True,
                           metavar='/path/to/my_prior_res',
                           help='Absolute file path of prior pRF results. \
                                 Ignored if in testing mode.'
                           )

    # Add argument to namespace -varNumOpt1 flag:
    objParser.add_argument('-varNumOpt1', required=True, type=int,
                           metavar='N1',
                           help='Number of radial positions.'
                           )

    # Add argument to namespace -varNumOpt2 flag:
    objParser.add_argument('-varNumOpt2', required=True, type=int,
                           metavar='N2',
                           help='Number of angular positions.'
                           )

    # Add argument to namespace -varNumOpt3 flag:
    objParser.add_argument('-varNumOpt3', default=None, metavar='N3',
                           help='Max displacement in radial direction.'
                           )

    # Add argument to namespace -lgcRstrCentre flag:
    objParser.add_argument('-lgcRstrCentre', dest='lgcRstrCentre',
                           action='store_true', default=False,
                           help='Restrict fitted models to stimulated area.')

    objParser.add_argument('-strPathHrf', default=None, required=False,
                           metavar='/path/to/custom_hrf_parameter.npy',
                           help='Path to npy file with custom hrf parameters. \
                           Ignored if in testing mode.')

    objParser.add_argument('-supsur', nargs='+',
                           help='List of floats that represent the ratio of \
                                 size neg surround to size pos center.',
                           type=float, default=None)

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    return objNspc