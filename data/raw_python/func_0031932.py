def get_raw_gids(model_params):
    '''
    Reads text file containing gids of neuron populations as created within the
    NEST simulation. These gids are not continuous as in the simulation devices
    get created in between.
    '''
    gidfile = open(os.path.join(model_params.raw_nest_output_path,
                                model_params.GID_filename), 'r') 
    gids = [] 
    for l in gidfile :
        a = l.split()
        gids.append([int(a[0]),int(a[1])])
    return gids