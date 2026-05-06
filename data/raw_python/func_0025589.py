def process_args():
    """ 
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
                description=("A script which can be run to generate a LEMS "
                             "file to analyse the behaviour of channels in "
                             "NeuroML 2"))

    parser.add_argument('channelFiles', 
                        type=str,
                        nargs='+',
                        metavar='<NeuroML 2 Channel file>', 
                        help="Name of the NeuroML 2 file(s)")
                        
                        
    parser.add_argument('-v',
                        action='store_true',
                        default=DEFAULTS['v'],
                        help="Verbose output")
                        
    parser.add_argument('-minV', 
                        type=int,
                        metavar='<min v>',
                        default=DEFAULTS['minV'],
                        help="Minimum voltage to test (integer, mV), default: %smV"%DEFAULTS['minV'])
                        
    parser.add_argument('-maxV', 
                        type=int,
                        metavar='<max v>',
                        default=DEFAULTS['maxV'],
                        help="Maximum voltage to test (integer, mV), default: %smV"%DEFAULTS['maxV'])
                        
    parser.add_argument('-temperature', 
                        type=float,
                        metavar='<temperature>',
                        default=DEFAULTS['temperature'],
                        help="Temperature (float, celsius), default: %sdegC"%DEFAULTS['temperature'])
                        
    parser.add_argument('-duration', 
                        type=float,
                        metavar='<duration>',
                        default=DEFAULTS['duration'],
                        help="Duration of simulation in ms, default: %sms"%DEFAULTS['duration'])
                        
    parser.add_argument('-clampDelay', 
                        type=float,
                        metavar='<clamp delay>',
                        default=DEFAULTS['clampDelay'],
                        help="Delay before voltage clamp is activated in ms, default: %sms"%DEFAULTS['clampDelay'])
                        
    parser.add_argument('-clampDuration', 
                        type=float,
                        metavar='<clamp duration>',
                        default=DEFAULTS['clampDuration'],
                        help="Duration of voltage clamp in ms, default: %sms"%DEFAULTS['clampDuration'])
                        
    parser.add_argument('-clampBaseVoltage', 
                        type=float,
                        metavar='<clamp base voltage>',
                        default=DEFAULTS['clampBaseVoltage'],
                        help="Clamp base (starting/finishing) voltage in mV, default: %smV"%DEFAULTS['clampBaseVoltage'])
                        
    parser.add_argument('-stepTargetVoltage', 
                        type=float,
                        metavar='<step target voltage>',
                        default=DEFAULTS['stepTargetVoltage'],
                        help=("Voltage in mV through which to step voltage clamps, default: %smV"%DEFAULTS['stepTargetVoltage']))
                        
    parser.add_argument('-erev', 
                        type=float,
                        metavar='<reversal potential>',
                        default=DEFAULTS['erev'],
                        help="Reversal potential of channel for currents, default: %smV"%DEFAULTS['erev'])
                        
    parser.add_argument('-scaleDt', 
                        type=float,
                        metavar='<scale dt in generated LEMS>',
                        default=DEFAULTS['scaleDt'],
                        help="Scale dt in generated LEMS, default: %s"%DEFAULTS['scaleDt'])
                        
    parser.add_argument('-caConc', 
                        type=float,
                        metavar='<Ca2+ concentration>',
                        default=DEFAULTS['caConc'],
                        help=("Internal concentration of Ca2+ (float, "
                              "concentration in mM), default: %smM"%DEFAULTS['caConc']))
                              
                        
    parser.add_argument('-datSuffix', 
                        type=str,
                        metavar='<dat suffix>',
                        default=DEFAULTS['datSuffix'],
                        help="String to add to dat file names (before .dat)")
                        
    parser.add_argument('-norun',
                        action='store_true',
                        default=DEFAULTS['norun'],
                        help=("If used, just generate the LEMS file, "
                              "don't run it"))
                        
    parser.add_argument('-nogui',
                        action='store_true',
                        default=DEFAULTS['nogui'],
                        help=("Supress plotting of variables and only save "
                              "data to file"))
                        
    parser.add_argument('-html',
                        action='store_true',
                        default=DEFAULTS['html'],
                        help=("Generate a HTML page featuring the plots for the "
                              "channel"))
                        
    parser.add_argument('-md',
                        action='store_true',
                        default=DEFAULTS['md'],
                        help=("Generate a (GitHub flavoured) Markdown page featuring the plots for the "
                              "channel"))
                        
    parser.add_argument('-ivCurve',
                        action='store_true',
                        default=DEFAULTS['ivCurve'],
                        help=("Save currents through voltage clamp at each "
                              "level & plot current vs voltage for ion "
                              "channel"))
                        
                        
    return parser.parse_args()