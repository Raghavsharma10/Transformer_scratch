def sphere_average(dset,x,y,z,radius=1):
    '''returns a list of average values (one for each subbrick/time point) within the coordinate ``(x,y,z)`` (in RAI order) using a sphere of radius ``radius`` in ``dset``'''
    return_list = []
    if isinstance(dset,basestring):
        dset = [dset]
    for d in dset:
        return_list += [float(a) for a in subprocess.check_output(['3dmaskave','-q','-dball',str(x),str(y),str(z),str(radius),d],stderr=subprocess.PIPE).split()]
    return return_list