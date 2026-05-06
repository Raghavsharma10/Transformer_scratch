def calc_variances(params):
    '''
    This function calculates the variance of the sum signal and all population-resolved signals
    '''

    depth = params.electrodeParams['z']

    ############################
    ### CSD                  ###
    ############################
 
    for i, data_type in enumerate(['CSD','LFP']):
        if i % SIZE == RANK:
    
            f_out = h5py.File(os.path.join(params.savefolder, ana_params.analysis_folder,
                                           data_type + ana_params.fname_variances), 'w')
            f_out['depths']=depth
      
            for celltype in params.y:
                f_in = h5py.File(os.path.join(params.populations_path,
                                              '%s_population_%s' % (celltype,data_type) + '.h5' ))
                var = f_in['data'].value[:, ana_params.transient:].var(axis=1)
                f_in.close()
                f_out[celltype]= var
            
            f_in = h5py.File(os.path.join(params.savefolder, data_type + 'sum.h5' ))
            var= f_in['data'].value[:, ana_params.transient:].var(axis=1)
            f_in.close()
            f_out['sum']= var
        
            f_out.close()

    return