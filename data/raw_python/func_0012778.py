def newidfobject(self, key, aname='', defaultvalues=True, **kwargs):
        """
        Add a new idfobject to the model. If you don't specify a value for a
        field, the default value will be set.

        For example ::

            newidfobject("CONSTRUCTION")
            newidfobject("CONSTRUCTION",
                Name='Interior Ceiling_class',
                Outside_Layer='LW Concrete',
                Layer_2='soundmat')

        Parameters
        ----------
        key : str
            The type of IDF object. This must be in ALL_CAPS.
        aname : str, deprecated
            This parameter is not used. It is left there for backward
            compatibility.
        defaultvalues: boolean
            default is True. If True default values WILL be set. 
            If False, default values WILL NOT be set
        **kwargs
            Keyword arguments in the format `field=value` used to set the value
            of fields in the IDF object when it is created.

        Returns
        -------
        EpBunch object

        """
        
        obj = newrawobject(self.model, self.idd_info, 
                    key, block=self.block, defaultvalues=defaultvalues)
        abunch = obj2bunch(self.model, self.idd_info, obj)
        if aname:
            warnings.warn("The aname parameter should no longer be used.", UserWarning)
            namebunch(abunch, aname)
        self.idfobjects[key].append(abunch)
        for k, v in list(kwargs.items()):
            abunch[k] = v
        return abunch