def process_args():
    """ 
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A script for plotting files containing spike time data")
    
    parser.add_argument('spiketimeFiles', 
                        type=str, 
                        metavar='<spiketime file>', 
                        help='List of text file containing spike times', 
                        nargs='+')
                        
    parser.add_argument('-format', 
                        type=str,
                        metavar='<format>',
                        default=DEFAULTS['format'],
                        help='How the spiketimes are represented on each line of file:\n'+\
                             'id_t: id of cell, space(s)/tab(s), time of spike (default);\n'+\
                             't_id: time of spike, space(s)/tab(s), id of cell;\n'+\
                             'sonata: SONATA format HDF5 file containing spike times')
                             
    parser.add_argument('-rates', 
                        action='store_true',
                        default=DEFAULTS['rates'],
                        help='Show a plot of rates')
                        
    parser.add_argument('-showPlotsAlready', 
                        action='store_true',
                        default=DEFAULTS['show_plots_already'],
                        help='Show plots once generated')
                        
    parser.add_argument('-saveSpikePlotTo', 
                        type=str,
                        metavar='<spiketime plot filename>',
                        default=DEFAULTS['save_spike_plot_to'],
                        help='Name of file in which to save spiketime plot')
                        
    parser.add_argument('-rateWindow', 
                        type=int,
                        metavar='<rate window>',
                        default=DEFAULTS['rate_window'],
                        help='Window for rate calculation in ms')
                        
    parser.add_argument('-rateBins', 
                        type=int,
                        metavar='<rate bins>',
                        default=DEFAULTS['rate_bins'],
                        help='Number of bins for rate histogram')
                        
    return parser.parse_args()