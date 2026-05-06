def PhyDMSComprehensiveParser():
    """Returns *argparse.ArgumentParser* for ``phdyms_comprehensive`` script."""
    parser = ArgumentParserNoArgHelp(description=("Comprehensive phylogenetic "
            "model comparison and detection of selection informed by deep "
            "mutational scanning data. This program runs 'phydms' repeatedly "
            "to compare substitution models and detect selection. "
            "{0} Version {1}. Full documentation at {2}").format(
            phydmslib.__acknowledgments__, phydmslib.__version__,
            phydmslib.__url__),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outprefix', help='Output file prefix.', type=str)
    parser.add_argument('alignment', help='Existing FASTA file with aligned '
            'codon sequences.', type=ExistingFile)
    parser.add_argument('prefsfiles', help='Existing files with site-specific '
            'amino-acid preferences.', type=ExistingFile, nargs='+')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--raxml', help="Path to RAxML (e.g., 'raxml')")
    group.add_argument('--tree', type=ExistingFile,
             help="Existing Newick file giving input tree.")
    parser.add_argument('--ncpus', default=-1, help='Use this many CPUs; -1 '
            'means all available.', type=int)
    parser.add_argument('--brlen', choices=['scale', 'optimize'],
            default='optimize', help=("How to handle branch lengths: "
            "scale by single parameter or optimize each one"))
    parser.set_defaults(omegabysite=False)
    parser.add_argument('--omegabysite', dest='omegabysite',
            action='store_true', help="Fit omega (dN/dS) for each site.")
    parser.set_defaults(diffprefsbysite=False)
    parser.add_argument('--diffprefsbysite', dest='diffprefsbysite',
            action='store_true', help="Fit differential preferences for "
            "each site.")
    parser.set_defaults(gammaomega=False)
    parser.add_argument('--gammaomega', dest='gammaomega', action=\
            'store_true', help="Fit ExpCM with gamma distributed omega.")
    parser.set_defaults(gammabeta=False)
    parser.add_argument('--gammabeta', dest='gammabeta', action=\
            'store_true', help="Fit ExpCM with gamma distributed beta.")
    parser.set_defaults(noavgprefs=False)
    parser.add_argument('--no-avgprefs', dest='noavgprefs', action='store_true',
            help="No fitting of models with preferences averaged across sites "
            "for ExpCM.")
    parser.set_defaults(randprefs=False)
    parser.add_argument('--randprefs', dest='randprefs', action='store_true',
            help="Include ExpCM models with randomized preferences.")
    parser.add_argument('-v', '--version', action='version', version=
            '%(prog)s {version}'.format(version=phydmslib.__version__))
    return parser