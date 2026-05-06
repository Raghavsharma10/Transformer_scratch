def get_build_params(metadata):
    '''get_build_params uses get_build_metadata to retrieve corresponding meta data values for a build
    :param metadata: a list, each item a dictionary of metadata, in format:
    metadata = [{'key': 'repo_url', 'value': repo_url },
                {'key': 'repo_id', 'value': repo_id },
                {'key': 'credential', 'value': credential },
                {'key': 'response_url', 'value': response_url },
                {'key': 'token', 'value': token},
                {'key': 'commit', 'value': commit }]

    '''
    params = dict()
    for item in metadata:
        if item['value'] == None:
            response = get_build_metadata(key=item['key'])
            item['value'] = response
        params[item['key']] = item['value']
        if item['key'] not in ['token', 'secret', 'credential']:
            bot.info('%s is set to %s' %(item['key'],item['value']))        
    return params