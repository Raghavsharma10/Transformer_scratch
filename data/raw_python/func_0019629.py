def _parse_params(self, params=None):
        """
        Parse parameters.

        Combine default and user-defined parameters.
        """
        prm = self.default_params.copy()
        if params is not None: 
            prm.update(params)
 
        if prm["background_model"]:
            # Absolute path, just to be sure
            prm["background_model"] = os.path.abspath(prm["background_model"])
        else:
            if prm.get("organism", None):
                prm["background_model"] = os.path.join(
                        self.config.get_bg_dir(), 
                        "{}.{}.bg".format(
                            prm["organism"], 
                            "MotifSampler"))
            else:            
                raise Exception("No background specified for {}".format(self.name))
        
        prm["strand"] = 1
        if prm["single"]:
            prm["strand"] = 0
        
        tmp = NamedTemporaryFile(dir=self.tmpdir)
        prm["pwmfile"] = tmp.name

        tmp2  = NamedTemporaryFile(dir=self.tmpdir)
        prm["outfile"] = tmp2.name
 
        return prm