def _parse_params(self, params=None):
        """
        Parse parameters.

        Combine default and user-defined parameters.
        """
        prm = self.default_params.copy()
        if params is not None: 
            prm.update(params)
 
        if prm["background"]:
            # Absolute path, just to be sure
            prm["background"] =  os.path.abspath(prm["background"])
            prm["background"] = " --negSet {0} ".format(
                    prm["background"])
        
        prm["strand"] = ""
        if not prm["single"]:
            prm["strand"] = " --revcomp "

        return prm