def transfer_attributes_to_instrument(self, inst, strict_names=False):
        """Transfer non-standard attributes in Meta to Instrument object.

        Pysat's load_netCDF and similar routines are only able to attach
        netCDF4 attributes to a Meta object. This routine identifies these
        attributes and removes them from the Meta object. Intent is to 
        support simple transfers to the pysat.Instrument object.

        Will not transfer names that conflict with pysat default attributes.
        
        Parameters
        ----------
        inst : pysat.Instrument
            Instrument object to transfer attributes to
        strict_names : boolean (False)
            If True, produces an error if the Instrument object already
            has an attribute with the same name to be copied.

        Returns
        -------
        None
            pysat.Instrument object modified in place with new attributes
        """

        # base Instrument attributes
        banned = inst._base_attr
        # get base attribute set, and attributes attached to instance
        base_attrb = self._base_attr
        this_attrb = dir(self)
        # collect these attributes into a dict
        adict = {}
        transfer_key = []
        for key in this_attrb:
            if key not in banned:
                if key not in base_attrb:
                    # don't store _ leading attributes
                    if key[0] != '_':
                        adict[key] = self.__getattribute__(key)
                        transfer_key.append(key)

        # store any non-standard attributes in Instrument
        # get list of instrument objects attributes first
        # to check if a duplicate
        inst_attr = dir(inst)
        for key in transfer_key:
            if key not in banned:
                if key not in inst_attr:
                    inst.__setattr__(key, adict[key])
                else:
                    if not strict_names:
                        # new_name = 'pysat_attr_'+key
                        inst.__setattr__(key, adict[key])
                    else:
                        raise RuntimeError('Attribute ' + key +
                                           'attached to Meta object can not be '
                                           + 'transferred as it already exists'
                                           + ' in the Instrument object.')