def PhyDMSLogoPlotParser():
    """Returns `argparse.ArgumentParser` for ``phydms_logoplot``."""
    parser = ArgumentParserNoArgHelp(description=
            "Make logo plot of preferences or differential preferences. "
            "Uses weblogo (http://weblogo.threeplusone.com/). "
            "{0} Version {1}. Full documentation at {2}".format(
            phydmslib.__acknowledgments__,
            phydmslib.__version__, phydmslib.__url__),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prefs', type=ExistingFile, help="File with "
            "amino-acid preferences; same format as input to 'phydms'.")
    group.add_argument('--diffprefs', type=ExistingFile, help="File with "
            "differential preferences; in format output by 'phydms'.")
    parser.add_argument('outfile', help='Name of created PDF logo plot.')
    parser.add_argument('--stringency', type=FloatGreaterThanEqualToZero,
            default=1, help="Stringency parameter to re-scale prefs.")
    parser.add_argument('--nperline', type=IntGreaterThanZero, default=70,
            help="Number of sites per line.")
    parser.add_argument('--numberevery', type=IntGreaterThanZero, default=10,
            help="Number sites at this interval.")
    parser.add_argument('--mapmetric', default='functionalgroup', choices=['kd',
            'mw', 'charge', 'functionalgroup'], help='Metric used to color '
            'amino-acid letters. kd = Kyte-Doolittle hydrophobicity; '
            'mw = molecular weight; functionalgroup = divide in 7 '
            'groups; charge = charge at neutral pH.')
    parser.add_argument('--colormap', type=str, default='jet',
            help="Name of `matplotlib` color map for amino acids "
            "when `--mapmetric` is 'kd' or 'mw'.")
    parser.add_argument('--diffprefheight', type=FloatGreaterThanZero,
            default=1.0, help="Height of diffpref logo in each direction.")
    parser.add_argument('--omegabysite', help="Overlay omega on "
            "logo plot. Specify '*_omegabysite.txt' file from 'phydms'.",
            type=ExistingFileOrNone)
    parser.add_argument('--minP', type=FloatGreaterThanZero, default=1e-4,
            help="Min plotted P-value for '--omegabysite' overlay.")
    parser.add_argument('-v', '--version', action='version',
            version='%(prog)s {version}'.format(version=phydmslib.__version__))
    return parser