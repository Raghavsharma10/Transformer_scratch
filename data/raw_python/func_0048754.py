def dictToH5Group(d, group, link_copy=True):
    """ helper function that transform (recursive) a dictionary into an
        hdf group by creating subgroups 
        link_copy = True, tries to save space in the hdf file by creating an internal link.
                    the current implementation uses memory though ...
    """
    for key in d.keys():
        value = d[key]
        log.debug("saving",key,"in",group)
        # hope for the best (i.e. h5py can handle that)
        try:
            if link_copy and isinstance(value,np.ndarray):
              value=_find_link(value,group,key)
            else:
              if isinstance(value,h5py.Dataset): value = value[:]; # hdf5 dataset have to be read first ...
            group[key] = value
        except (TypeError,ValueError) as e:
            log.debug("For %s, h5py could not handle the saving on its own, trying to convert it, error was %s"%(key,e))
            if isinstance(value,dict) or hasattr(value,"__dict__"):
                if key not in group: group.create_group(key)
                try:
                    value = dictToH5Group(value,group[key],link_copy=link_copy)
                # objects have __dict__ but can be coverted to dict like only 
                # by DataStorage (and not by dict)
                except:
                    value = dictToH5Group(DataStorage(value),group[key],link_copy=link_copy)
            # take care of unicode (h5py can't handle numpy unicode arrays)
            elif isinstance(value,np.ndarray) and value.dtype.char == "U":
                value = np.asarray([vv.encode('ascii') for vv in value])
                group[key] = value
            elif isinstance(value, collections.Iterable):
                if key not in group: group.create_group(key)
                group[key].attrs["IS_LIST"] = True
                fmt = "index%%0%dd" % math.ceil(np.log10(len(value)))
                for index, array in enumerate(value):
                    dictToH5Group({fmt % index: array},
                              group[key], link_copy=link_copy)
            elif value is None:
                group[key] = "NONE_PYTHON_OBJECT"
            else:
                log.warn("Could not convert %s into an object that can be saved"%key)