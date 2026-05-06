def load_files(files, tag=None, sat_id=None):
    '''Loads a list of COSMIC data files, supplied by user.
    
    Returns a list of dicts, a dict for each file.
    '''
       
    output = [None]*len(files)
    drop_idx = []
    for (i,file) in enumerate(files):
        try:
            data = netCDF4.Dataset(file)    
            # build up dictionary will all ncattrs
            new = {} 
            # get list of file attributes
            ncattrsList = data.ncattrs()
            for d in ncattrsList:
                new[d] = data.getncattr(d)            
            # load all of the variables in the netCDF
            loadedVars={}
            keys = data.variables.keys()
            for key in keys:
                loadedVars[key] = data.variables[key][:]               
            new['profiles'] = pysat.DataFrame(loadedVars)
            if tag == 'ionprf':
                new['profiles'].index = new['profiles']['MSL_alt']
            output[i] = new   
            data.close()
        except RuntimeError:
            # some of the S4 files have zero bytes, which causes a read error
            # this stores the index of these zero byte files so I can drop 
            # the Nones the gappy file leaves behind
            drop_idx.append(i)

    # drop anything that came from the zero byte files
    drop_idx.reverse()
    for i in drop_idx:
        del output[i]
    return output