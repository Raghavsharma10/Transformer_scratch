def process_args():
    """ 
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
                description=("A script which can be run to tune a NeuroML 2 model against a number of target properties. Work in progress!"))

                        
    parser.add_argument('prefix', 
                        type=str,
                        metavar='<prefix>', 
                        help="Prefix for optimisation run")

    parser.add_argument('neuromlFile', 
                        type=str,
                        metavar='<neuromlFile>', 
                        help="NeuroML2 file containing model")

    parser.add_argument('target', 
                        type=str,
                        metavar='<target>', 
                        help="Target in NeuroML2 model")
                        
    parser.add_argument('parameters', 
                        type=str,
                        metavar='<parameters>', 
                        help="List of parameter to adjust")
                        
    parser.add_argument('maxConstraints', 
                        type=str,
                        metavar='<max_constraints>', 
                        help="Max values for parameters")
                        
    parser.add_argument('minConstraints', 
                        type=str,
                        metavar='<min_constraints>', 
                        help="Min values for parameters")
                        
    parser.add_argument('targetData', 
                        type=str,
                        metavar='<targetData>', 
                        help="List of name/value pairs for properties extracted from data to judge fitness against")
                        
    parser.add_argument('weights', 
                        type=str,
                        metavar='<weights>', 
                        help="Weights to assign to each target name/value pair")
                        
    parser.add_argument('-simTime', 
                        type=float,
                        metavar='<simTime>', 
                        default=DEFAULTS['simTime'],
                        help="Simulation duration")
                        
    parser.add_argument('-dt', 
                        type=float,
                        metavar='<dt>', 
                        default=DEFAULTS['dt'],
                        help="Simulation timestep")
                        
    parser.add_argument('-analysisStartTime', 
                        type=float,
                        metavar='<analysisStartTime>', 
                        default=DEFAULTS['analysisStartTime'],
                        help="Analysis start time")
                        
    parser.add_argument('-populationSize', 
                        type=int,
                        metavar='<populationSize>', 
                        default=DEFAULTS['populationSize'],
                        help="Population size")
                        
    parser.add_argument('-maxEvaluations', 
                        type=int,
                        metavar='<maxEvaluations>', 
                        default=DEFAULTS['maxEvaluations'],
                        help="Maximum evaluations")
                        
    parser.add_argument('-numSelected', 
                        type=int,
                        metavar='<numSelected>', 
                        default=DEFAULTS['numSelected'],
                        help="Number selected")
                        
    parser.add_argument('-numOffspring', 
                        type=int,
                        metavar='<numOffspring>', 
                        default=DEFAULTS['numOffspring'],
                        help="Number offspring")
                        
            
    parser.add_argument('-mutationRate', 
                        type=float,
                        metavar='<mutationRate>', 
                        default=DEFAULTS['mutationRate'],
                        help="Mutation rate")
                        
    parser.add_argument('-numElites', 
                        type=int,
                        metavar='<numElites>', 
                        default=DEFAULTS['numElites'],
                        help="Number of elites")
                        
    parser.add_argument('-numParallelEvaluations', 
                        type=int,
                        metavar='<numParallelEvaluations>', 
                        default=DEFAULTS['numParallelEvaluations'],
                        help="Number of evaluations to run in parallel")
                        
    parser.add_argument('-seed', 
                        type=int,
                        metavar='<seed>', 
                        default=DEFAULTS['seed'],
                        help="Seed for optimiser")
                        
    parser.add_argument('-simulator', 
                        type=str,
                        metavar='<simulator>', 
                        default=DEFAULTS['simulator'],
                        help="Simulator to run")
                        
    parser.add_argument('-knownTargetValues', 
                        type=str,
                        metavar='<knownTargetValues>', 
                        help="List of name/value pairs which represent the known values of the target parameters")
                        
    parser.add_argument('-nogui', 
                        action='store_true',
                        default=DEFAULTS['nogui'],
                        help="Should GUI elements be supressed?")
                        
    parser.add_argument('-showPlotAlready', 
                        action='store_true',
                        default=DEFAULTS['showPlotAlready'],
                        help="Should generated plots be suppressed until show() called?")
                        
    parser.add_argument('-verbose', 
                        action='store_true',
                        default=DEFAULTS['verbose'],
                        help="Verbose mode")
                        
    parser.add_argument('-dryRun', 
                        action='store_true',
                        default=DEFAULTS['dryRun'],
                        help="Dry run; just print setup information")
                        
    parser.add_argument('-extraReportInfo', 
                        type=str,
                        metavar='<extraReportInfo>', 
                        default=DEFAULTS['extraReportInfo'],
                        help='Extra tag/value pairs can be put into the report.json:  -extraReportInfo=["tag":"value"]')
                        
    parser.add_argument('-cleanup', 
                        action='store_true',
                        default=DEFAULTS['cleanup'],
                        help="Should (some) generated files, e.g. *.dat, be deleted as optimisation progresses?")
                        
    return parser.parse_args()