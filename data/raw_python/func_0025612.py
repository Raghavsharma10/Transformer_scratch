def process_args():
    """ 
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A file for converting NeuroML v2 files into POVRay files for 3D rendering")

    parser.add_argument('neuroml_file', type=str, metavar='<NeuroML file>', 
                        help='NeuroML (version 2 beta 3+) file to be converted to PovRay format (XML or HDF5 format)')
                        
    parser.add_argument('-split',
                        action='store_true',
                        default=False,
                        help="If this is specified, generate separate pov files for cells & network. Default is false")


    parser.add_argument('-background', 
                        type=str,
                        metavar='<background colour>',
                        default=_WHITE,
                        help='Colour of background, e.g. <0,0,0,0.55>')

    parser.add_argument('-movie',
                        action='store_true',
                        default=False,
                        help="If this is specified, generate a ini file for generating a sequence of frames for a movie of the 3D structure")
                        
    parser.add_argument('-inputs',
                        action='store_true',
                        default=False,
                        help="If this is specified, show the locations of (synaptic, current clamp, etc.) inputs into the cells of the network")
                        
    parser.add_argument('-conns',
                        action='store_true',
                        default=False,
                        help="If this is specified, show the connections present in the network with lines")
                        
    parser.add_argument('-conn_points',
                        action='store_true',
                        default=False,
                        help="If this is specified, show the end points of the connections present in the network")
                        
    parser.add_argument('-v',
                        action='store_true',
                        default=False,
                        help="Verbose output")

    parser.add_argument('-frames', 
                        type=int,
                        metavar='<frames>',
                        default=36,
                        help='Number of frames in movie')
                        
    parser.add_argument('-posx', 
                        type=float,
                        metavar='<position offset x>',
                        default=0,
                        help='Offset position in x dir (0 is centre, 1 is top)')
    parser.add_argument('-posy', 
                        type=float,
                        metavar='<position offset y>',
                        default=0,
                        help='Offset position in y dir (0 is centre, 1 is top)')
    parser.add_argument('-posz', 
                        type=float,
                        metavar='<position offset z>',
                        default=0,
                        help='Offset position in z dir (0 is centre, 1 is top)')
                        
    parser.add_argument('-viewx', 
                        type=float,
                        metavar='<view offset x>',
                        default=0,
                        help='Offset viewing point in x dir (0 is centre, 1 is top)')
    parser.add_argument('-viewy', 
                        type=float,
                        metavar='<view offset y>',
                        default=0,
                        help='Offset viewing point in y dir (0 is centre, 1 is top)')
    parser.add_argument('-viewz', 
                        type=float,
                        metavar='<view offset z>',
                        default=0,
                        help='Offset viewing point in z dir (0 is centre, 1 is top)')

    parser.add_argument('-scalex', 
                        type=float,
                        metavar='<scale position x>',
                        default=1,
                        help='Scale position from network in x dir')
    parser.add_argument('-scaley', 
                        type=float,
                        metavar='<scale position y>',
                        default=1.5,
                        help='Scale position from network in y dir')
    parser.add_argument('-scalez', 
                        type=float,
                        metavar='<scale position z>',
                        default=1,
                        help='Scale position from network in z dir')

    parser.add_argument('-mindiam', 
                        type=float,
                        metavar='<minimum diameter dendrites/axons>',
                        default=0,
                        help='Minimum diameter for dendrites/axons (to improve visualisations)')

    parser.add_argument('-plane',
                        action='store_true',
                        default=False,
                        help="If this is specified, add a 2D plane below cell/network")

    parser.add_argument('-segids',
                        action='store_true',
                        default=False,
                        help="Show segment ids")
    
    return parser.parse_args()