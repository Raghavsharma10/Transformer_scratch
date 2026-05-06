def get_L_yXL(fname, y, x_in_X, L):
    '''
    compute the layer specificity, defined as:
    ::
    
        L_yXL = k_yXL / k_yX
    '''
    def _get_L_yXL_per_yXL(fname, x_in_X, X_index,
                                  y, layer):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        
        # Get number of synapses
        if layer in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
            #init variables
            k_yXL = 0
            k_yX = 0
            
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][layer][x] / 100.
                k_yL = data['data'][y]['syn_dict'][layer]['number of synapses per neuron']
                k_yXL += p_yxL * k_yL
                
            for l in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
                for x in x_in_X[X_index]:
                    p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                    k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                    k_yX +=  p_yxL * k_yL
            
            if k_yXL != 0.:
                return k_yXL / k_yX
            else:
                return 0.
        else:
            return 0.


    #init dict
    L_yXL = {}

    #iterate over postsynaptic cell types
    for y_value in y:
        #container
        data = np.zeros((len(L), len(x_in_X)))
        #iterate over lamina
        for i, Li in enumerate(L):
            #iterate over presynapse population inds
            for j in range(len(x_in_X)):
                data[i][j]= _get_L_yXL_per_yXL(fname, x_in_X,
                                                          X_index=j,
                                                          y=y_value,
                                                          layer=Li)
        L_yXL[y_value] = data

    return L_yXL