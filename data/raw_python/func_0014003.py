def _label_setter(self, new_label, current_label, attr_label, default=np.NaN, use_names_default=False):
        """Generalized setter of default meta attributes
        
        Parameters
        ----------
        new_label : str
            New label to use in the Meta object
        current_label : str
            The hidden attribute to be updated that actually stores metadata
        default : 
            Deafult setting to use for label if there is no attribute
            value
        use_names_default : bool
            if True, MetaData variable names are used as the default
            value for the specified Meta attributes settings
            
        Examples
        --------
        :
                @name_label.setter   
                def name_label(self, new_label):
                    self._label_setter(new_label, self._name_label, 
                                        use_names_default=True)  
        
        Notes
        -----
        Not intended for end user
                                  
        """
        
        if new_label not in self.attrs():
            # new label not in metadata, including case
            # update existing label, if present
            if current_label in self.attrs():
                # old label exists and has expected case
                self.data.loc[:, new_label] = self.data.loc[:, current_label]
                self.data.drop(current_label, axis=1, inplace=True)
            else:
                if self.has_attr(current_label):
                    # there is something like label, wrong case though
                    current_label = self.attr_case_name(current_label)
                    self.data.loc[:, new_label] = self.data.loc[:, current_label]
                    self.data.drop(current_label, axis=1, inplace=True)
                else:
                    # there is no existing label
                    # setting for the first time
                    if use_names_default:
                        self.data[new_label] = self.data.index
                    else:
                        self.data[new_label] = default
            # check higher order structures as well
            # recursively change labels here
            for key in self.keys_nD():
                setattr(self.ho_data[key], attr_label, new_label)

        # now update 'hidden' attribute value
        # current_label = new_label
        setattr(self, ''.join(('_',attr_label)), new_label)