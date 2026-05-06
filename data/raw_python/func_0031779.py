def get_T_yX(fname, y, y_in_Y, x_in_X, F_y):
    '''
    compute the cell type specificity, defined as:
    ::
    
        T_yX = K_yX / K_YX
            = F_y * k_yX / sum_y(F_y*k_yX) 
    
    '''
    def _get_k_yX_mul_F_y(y, y_index, X_index):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        #init variables
        k_yX = 0.
        
        for l in [str(key) for key in data['data'][y]['syn_dict'].keys()]:
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                k_yX +=  p_yxL * k_yL
        
        return k_yX * F_y[y_index]


    #container
    T_yX = np.zeros((len(y), len(x_in_X)))
    
    #iterate over postsynaptic cell types
    for i, y_value in enumerate(y):
        #iterate over presynapse population inds
        for j in range(len(x_in_X)):
            k_yX_mul_F_y = 0
            for k, yy in enumerate(sum(y_in_Y, [])):                
                if y_value in yy:
                    for yy_value in yy:
                        ii = np.where(np.array(y) == yy_value)[0][0]
                        k_yX_mul_F_y += _get_k_yX_mul_F_y(yy_value, ii, j)
            
            
            if k_yX_mul_F_y != 0:
                T_yX[i, j] = _get_k_yX_mul_F_y(y_value, i, j) / k_yX_mul_F_y
            
    return T_yX