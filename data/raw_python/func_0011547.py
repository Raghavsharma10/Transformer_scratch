def get_runtime_vars(varset, experiment, token):
    '''get_runtime_vars will return the urlparsed string of one or more runtime
       variables. If None are present, None is returned.
  
       Parameters
       ==========
       varset: the variable set, a dictionary lookup with exp_id, token, vars
       experiment: the exp_id to look up
       token: the participant id (or token) that must be defined.
 
       Returns
       =======
       url: the variable portion of the url to be passed to experiment, e.g,
            '?words=at the thing&color=red&globalname=globalvalue'

    '''
    url = ''
    if experiment in varset:

        variables = dict()

        # Participant set variables

        if token in varset[experiment]:
            for k,v in varset[experiment][token].items():
                variables[k] = v

        # Global set variables
        if "*" in varset[experiment]:
            for k,v in varset[experiment]['*'].items():

                # Only add the variable if not already defined
                if k not in variables:
                    variables[k] = v

        # Join together, the first ? is added by calling function
        varlist = ["%s=%s" %(k,v) for k,v in variables.items()]
        url = '&'.join(varlist)

    bot.debug('Parsed url: %s' %url)
    return url