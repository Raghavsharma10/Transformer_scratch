def PhyDMSParser():
    """Returns *argparse.ArgumentParser* for ``phydms`` script."""
    parser = ArgumentParserNoArgHelp(description=('Phylogenetic analysis '
            'informed by deep mutational scanning data. {0} Version {1}. Full'
            ' documentation at {2}').format(phydmslib.__acknowledgments__,
            phydmslib.__version__, phydmslib.__url__),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('alignment', type=ExistingFile,
            help='Existing FASTA file with aligned codon sequences.')
    parser.add_argument('tree', type=ExistingFile,
            help="Existing Newick file giving input tree.")
    parser.add_argument('model', type=ModelOption,
            help=("Substitution model: ExpCM_<prefsfile> or YNGKP_<m> ("
            "where <m> is {0}). For ExpCM, <prefsfile> has first "
            "column labeled 'site' and others labeled by 1-letter "
            "amino-acid code.").format(', '.join(yngkp_modelvariants)))
    parser.add_argument('outprefix', help='Output file prefix.', type=str)
    parser.add_argument('--brlen', choices=['scale', 'optimize'],
            default='optimize', help=("How to handle branch lengths: "
            "scale by single parameter or optimize each one"))
    parser.set_defaults(gammaomega=False)
    parser.add_argument('--gammaomega', action='store_true',
            dest='gammaomega', help="Omega for ExpCM from gamma "
            "distribution rather than single value. To achieve "
            "same for YNGKP, use 'model' of YNGKP_M5.")
    parser.set_defaults(gammabeta=False)
    parser.add_argument('--gammabeta', action='store_true',
            dest='gammabeta', help="Beta for ExpCM from gamma "
            "distribution rather than single value.")
    parser.set_defaults(omegabysite=False)
    parser.add_argument('--omegabysite', dest='omegabysite',
            action='store_true', help="Fit omega (dN/dS) for each site.")
    parser.set_defaults(omegabysite_fixsyn=False)
    parser.add_argument('--omegabysite_fixsyn', dest='omegabysite_fixsyn',
            action='store_true', help="For '--omegabysite', assign all "
            "sites same dS rather than fit for each site.")
    parser.set_defaults(diffprefsbysite=False)
    parser.add_argument('--diffprefsbysite', dest='diffprefsbysite',
            action='store_true', help="Fit differential preferences "
            "for each site.")
    parser.add_argument('--diffprefsprior', default='invquadratic,150,0.5',
            type=diffPrefsPrior, help="Regularizing prior for "
            "'--diffprefsbysite': 'invquadratic,C1,C2' is prior in "
            "Bloom, Biology Direct, 12:1.")
    parser.set_defaults(fitphi=False)
    parser.add_argument('--fitphi', action='store_true', dest='fitphi',
            help='Fit ExpCM phi rather than setting so stationary '
            'state matches alignment frequencies.')
    parser.set_defaults(randprefs=False)
    parser.add_argument('--randprefs', dest='randprefs', action='store_true',
            help="Randomize preferences among sites for ExpCM.")
    parser.set_defaults(avgprefs=False)
    parser.add_argument('--avgprefs', dest='avgprefs', action='store_true',
            help="Average preferences across sites for ExpCM.")
    parser.add_argument('--divpressure', type=ExistingFileOrNone,
            help=("Known diversifying pressure at sites: file with column 1 "
            "= position, column 2 = diversification pressure; columns space-, "
            "tab-, or comma-delimited."))
    parser.add_argument('--ncpus', default=1, type=int,
            help='Use this many CPUs; -1 means all available.')
    parser.add_argument('--fitprefsmethod', choices=[1, 2], default=2,
            help='Implementation to we use when fitting prefs.', type=int)
    parser.add_argument('--ncats', default=4, type=IntGreaterThanOne,
            help='Number of categories for gamma-distribution.')
    parser.add_argument('--minbrlen', type=FloatGreaterThanZero,
            default=phydmslib.constants.ALMOST_ZERO,
            help="Adjust all branch lengths in starting 'tree' to >= this.")
    parser.add_argument('--minpref', default=0.002, type=FloatGreaterThanZero,
            help="Adjust all preferences in ExpCM 'prefsfile' to >= this.")
    parser.add_argument('--seed', type=int, default=1, help="Random number seed.")
    parser.add_argument('--initparams', type=ExistingFile, help="Initialize "
            "model params from this file, which should be format of "
            "'*_modelparams.txt' file created by 'phydms' with this model.")
    parser.set_defaults(profile=False)
    parser.add_argument('--profile', dest='profile', action='store_true',
            help="Profile likelihood maximization, write pstats files. "
            "For code-development purposes.")
    parser.set_defaults(opt_details=False)
    parser.add_argument('--opt_details', dest='opt_details',
            action='store_true', help='Print details about optimization')
    parser.set_defaults(nograd=False)
    parser.add_argument('--nograd', dest='nograd', action='store_true',
            help="Do not use gradients for likelihood maximization.")
    parser.add_argument('-v', '--version', action='version', version=(
            ('%(prog)s {version}'.format(version=phydmslib.__version__))))
    return parser