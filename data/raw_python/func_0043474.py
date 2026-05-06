def get_config( config_path=CONFIG_PATH ):
    """
    Get the config
    """
   
    parser = SafeConfigParser()
    parser.read( config_path )

    config_dir = os.path.dirname(config_path)

    immutable_key = False
    key_id = None
    blockchain_id = None
    hostname = socket.gethostname()
    wallet = None
 
    if parser.has_section('blockstack-file'):

        if parser.has_option('blockstack-file', 'immutable_key'):
            immutable_key = parser.get('blockstack-file', 'immutable_key')
            if immutable_key.lower() in ['1', 'yes', 'true']:
                immutable_key = True
            else:
                immutable_key = False

        if parser.has_option('blockstack-file', 'file_id'):
            key_id = parser.get('blockstack-file', 'key_id' )

        if parser.has_option('blockstack-file', 'blockchain_id'):
            blockchain_id = parser.get('blockstack-file', 'blockchain_id')

        if parser.has_option('blockstack-file', 'hostname'):
            hostname = parser.get('blockstack-file', 'hostname')

        if parser.has_option('blockstack-file', 'wallet'):
            wallet = parser.get('blockstack-file', 'wallet')
        
    config = {
        'immutable_key': immutable_key,
        'key_id': key_id,
        'blockchain_id': blockchain_id,
        'hostname': hostname,
        'wallet': wallet
    }

    return config