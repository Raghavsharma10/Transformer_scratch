def make_lookup(experiment_list, key='exp_id'):
    '''make_lookup returns dict object to quickly look up query experiment on exp_id
    :param experiment_list: a list of query (dict objects)
    :param key_field: the key in the dictionary to base the lookup key (str)
    :returns lookup: dict (json) with key as "key_field" from query_list 
    '''
    lookup = dict()
    for single_experiment in experiment_list:
        if isinstance(single_experiment, str):
            single_experiment = load_experiment(single_experiment)
        lookup_key = single_experiment[key]
        lookup[lookup_key] = single_experiment
    return lookup