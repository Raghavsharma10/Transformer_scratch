def model_inspect(obj):
        '''
        Analize itself looking for special information, right now it returns:
        - Application name
        - Model name
        '''
        # Prepare the information object
        info = {}
        if hasattr(obj, '_meta'):
            info['verbose_name'] = getattr(obj._meta, 'verbose_name', None)
        else:
            info['verbose_name'] = None

        # Get info from the object
        if hasattr(obj, 'model') and obj.model:
            model = obj.model
        else:
            model = obj.__class__

        namesp = str(model)
        namesp = namesp.replace("<class ", "").replace(">", "").replace("'", "").split(".")

        # Remember information
        info['appname'] = namesp[-3]
        info['modelname'] = namesp[-1]
        info['model'] = model

        # Return the info
        return info