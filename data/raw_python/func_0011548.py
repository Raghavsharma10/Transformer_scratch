def generate_runtime_vars(variable_file=None, sep=','):
    '''generate a lookup data structure from a 
       delimited file. We typically obtain the file name and delimiter from
       the environment by way of EXPFACTORY_RUNTIME_VARS, and
       EXPFACTORY_RUNTIME_DELIM, respectively, but the user can also parse
       from a custom variable file by way of specifying it to the function
       (preference is given here). The file should be csv, with the
       only required first header field as "token" and second as "exp_id" to
       distinguish the participant ID and experiment id. The subsequent
       columns should correspond to experiment variable names. No special parsing
       of either is done. 

       Parameters
       ==========
       variable_file: full path to the tabular file with token, exp_id, etc.
       sep: the default delimiter to use, if not set in enironment.

       Returns
       =======
       varset: a dictionary lookup by exp_id and then participant ID.

       { 'test-parse-url': {
                             '123': {
                                      'color': 'red',
                                      'globalname': 'globalvalue',
                                      'words': 'at the thing'
                                    },

                             '456': {'color': 'blue',
                                     'globalname': 'globalvalue',
                                     'words': 'omg tacos'}
                              }
       }

    '''

    # First preference goes to runtime, then environment, then unset

    if variable_file is None:    
        if EXPFACTORY_RUNTIME_VARS is not None:
            variable_file = EXPFACTORY_RUNTIME_VARS

    if variable_file is not None:
        if not os.path.exists(variable_file):
            bot.warning('%s is set, but not found' %variable_file)
            return variable_file

    # If still None, no file
    if variable_file is None:
        return variable_file

    # If we get here, we have a variable file that exists
    delim = sep
    if EXPFACTORY_RUNTIME_DELIM is not None:
        delim = EXPFACTORY_RUNTIME_DELIM
    bot.debug('Delim for variables file set to %s' %sep)

    # Read in the file, generate config

    varset = dict()
    rows = _read_runtime_vars(variable_file)
    
    if len(rows) > 0:

        # When we get here, we are sure to have 
        # 'exp_id', 'var_name', 'var_value', 'token'

        for row in rows:

            exp_id = row[0].lower()   # exp-id must be lowercase
            var_name = row[1]
            var_value = row[2]
            token = row[3]

            # Level 1: Experiment ID
            if exp_id not in varset:
                varset[exp_id] = {}

            # Level 2: Participant ID
            if token not in varset[exp_id]:
                varset[exp_id][token] = {}

            # If found global setting, courtesy debug message
            if token == "*":
                bot.debug('Found global variable %s' %var_name)

            # Level 3: is the variable, issue warning if already defined
            if var_name in varset[exp_id][token]:
                bot.warning('%s defined twice %s:%s' %(var_name, exp_id, token))
            varset[exp_id][token][var_name] = var_value


    return varset