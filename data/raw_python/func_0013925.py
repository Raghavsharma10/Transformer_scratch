def load(cosmicFiles, tag=None, sat_id=None):
    """
    cosmic data load routine, called by pysat
    """   
    import netCDF4

    num = len(cosmicFiles)
    # make sure there are files to read
    if num != 0:
        # call separate load_files routine, segemented for possible
        # multiprocessor load, not included and only benefits about 20%
        output = pysat.DataFrame(load_files(cosmicFiles, tag=tag, sat_id=sat_id))
        output.index = pysat.utils.create_datetime_index(year=output.year, 
                month=output.month, day=output.day, 
                uts=output.hour*3600.+output.minute*60.+output.second)
        # make sure UTS strictly increasing
        output.sort_index(inplace=True)
        # use the first available file to pick out meta information
        meta = pysat.Meta()
        ind = 0
        repeat = True
        while repeat:
            try:
                data = netCDF4.Dataset(cosmicFiles[ind]) 
                ncattrsList = data.ncattrs()
                for d in ncattrsList:
                    meta[d] = {'units':'', 'long_name':d}
                keys = data.variables.keys()
                for key in keys:
                    meta[key] = {'units':data.variables[key].units, 
                                'long_name':data.variables[key].long_name}  
                repeat = False                  
            except RuntimeError:
                # file was empty, try the next one by incrementing ind
                ind+=1
                                    
        return output, meta
    else:
        # no data
        return pysat.DataFrame(None), pysat.Meta()