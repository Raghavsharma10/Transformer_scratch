def process_args():
    """ 
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A file for overlaying POVRay files generated from NeuroML by NeuroML1ToPOVRay.py with cell activity (e.g. as generated from a neuroConstruct simulation)")

    parser.add_argument('prefix', 
                        type=str, 
                        metavar='<network prefix>', 
                        help='Prefix for files in PovRay, e.g. use PREFIX is files are PREFIX.pov, PREFIX_net.inc, etc.')
                        
    parser.add_argument('-activity',
                        action='store_true',
                        default=False,
                        help="If this is specified, overlay network activity (not tested!!)")


    parser.add_argument('-maxV', 
                        type=float,
                        metavar='<maxV>',
                        default=50.0,
                        help='Max voltage for colour scale in mV')

    parser.add_argument('-minV', 
                        type=float,
                        metavar='<minV>',
                        default=-90.0,
                        help='Min voltage for colour scale in mV')

    parser.add_argument('-startTime', 
                        type=float,
                        metavar='<startTime>',
                        default=0,
                        help='Time in ms at which to start overlaying the simulation activity')
                        
    parser.add_argument('-endTime', 
                        type=float,
                        metavar='<endTime>',
                        default=100,
                        help='End time of simulation activity in ms')
                        
    parser.add_argument('-title', 
                        type=str, 
                        metavar='<title>', 
                        default='Movie generated from neuroConstruct simulation',
                        help='Title for movie')
                        
    parser.add_argument('-left', 
                        type=str, 
                        metavar='<left info>', 
                        default='',
                        help='Text on left')
                        
    parser.add_argument('-frames', 
                        type=int,
                        metavar='<frames>',
                        default=100,
                        help='Number of frames')
                        
    parser.add_argument('-name', 
                        type=str, 
                        metavar='<Movie name>', 
                        default='output',
                        help='Movie name')
                        
    return parser.parse_args()