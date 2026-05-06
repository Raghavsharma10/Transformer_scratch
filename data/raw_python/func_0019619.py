def _parse_params(self, params=None):
        """
        Parse parameters.

        Combine default and user-defined parameters.
        """
        prm = self.default_params.copy()
        if params is not None: 
            prm.update(params)
 
        # Background file is essential!
        if not prm["background"]:
            print("Background file needed!")
            sys.exit()
        
        prm["background"] =  os.path.abspath(prm["background"])
        
        prm["strand"] = ""
        if prm["single"]:
            prm["strand"] = " -strand + "
        
        return prm