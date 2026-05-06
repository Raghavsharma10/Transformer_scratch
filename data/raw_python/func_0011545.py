def _validate_config(self, folder, validate_folder=True):
        ''' validate config is the primary validation function that checks
            for presence and format of required fields.

        Parameters
        ==========
        :folder: full path to folder with config.json
        :name: if provided, the folder name to check against exp_id
        '''
        config = "%s/config.json" % folder
        name = os.path.basename(folder)
        if not os.path.exists(config):
            return notvalid("%s: config.json not found." %(folder))

        # Load the config
        try:
            config = read_json(config)
        except:
            return notvalid("%s: cannot load json, invalid." %(name))
 
        # Config.json should be single dict
        if isinstance(config, list):
            return notvalid("%s: config.json is a list, not valid." %(name))

        # Check over required fields
        fields = self.get_validation_fields()
        for field,value,ftype in fields:

            bot.verbose('field: %s, required: %s' %(field,value))

            # Field must be in the keys if required
            if field not in config.keys():
                if value == 1:
                    return notvalid("%s: config.json is missing required field %s" %(name,field))

            # Field is present, check type
            else:
                if not isinstance(config[field], ftype):
                    return notvalid("%s: invalid type, must be %s." %(name,str(ftype)))

            # Expid gets special treatment
            if field == "exp_id" and validate_folder is True:
                if config[field] != name:
                    return notvalid("%s: exp_id parameter %s does not match folder name." 
                                    %(name,config[field]))

                # name cannot have special characters, only _ and letters/numbers
                if not re.match("^[a-z0-9_-]*$", config[field]): 
                    message = "%s: exp_id parameter %s has invalid characters" 
                    message += "only lowercase [a-z],[0-9], -, and _ allowed."
                    return notvalid(message %(name,config[field]))
                

        return True