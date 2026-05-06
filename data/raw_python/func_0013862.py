def load_files(files, tag=None, sat_id=None, altitude_bin=None):
    '''Loads a list of COSMIC data files, supplied by user.
    
    Returns a list of dicts, a dict for each file.
    '''
       
    output = [None]*len(files)
    drop_idx = []
    for (i,file) in enumerate(files):
        try:
            #data = netCDF4.Dataset(file)    
            data = netcdf_file(file, mode='r', mmap=False) 
            # build up dictionary will all ncattrs
            new = {} 
            # get list of file attributes
            #ncattrsList = data.ncattrs()
            ncattrsList = data._attributes.keys()
            for d in ncattrsList:
                new[d] = data._attributes[d] #data.getncattr(d)
            # load all of the variables in the netCDF
            loadedVars={}
            keys = data.variables.keys()
            for key in keys:
                if data.variables[key][:].dtype.byteorder != '=':
                    loadedVars[key] = data.variables[key][:].byteswap().newbyteorder()
                else:
                    loadedVars[key] = data.variables[key][:]

            new['profiles'] = pysat.DataFrame(loadedVars)
                    
            output[i] = new   
            data.close()
        except RuntimeError:
            # some of the files have zero bytes, which causes a read error
            # this stores the index of these zero byte files so I can drop 
            # the Nones the gappy file leaves behind
            drop_idx.append(i)

    # drop anything that came from the zero byte files
    drop_idx.reverse()
    for i in drop_idx:
        del output[i]
        
    if tag == 'ionprf':           
        if altitude_bin is not None:
            for out in output:    
                out['profiles'].index = (out['profiles']['MSL_alt']/altitude_bin).round().values*altitude_bin
                out['profiles'] = out['profiles'].groupby(out['profiles'].index.values).mean()
        else:
            for out in output:
                out['profiles'].index = out['profiles']['MSL_alt']

    return output