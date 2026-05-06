def cleanup(connector_manager, red_data, tmp_dir):
    """
    Invokes the cleanup functions for all inputs.
    """
    for key, arg in red_data['inputs'].items():
        val = arg

        if isinstance(arg, list):
            for index, i in enumerate(arg):
                if not isinstance(i, dict):
                    continue

                # connector_class should be one of 'File' or 'Directory'
                connector_class = i['class']
                input_key = '{}_{}'.format(key, index)
                path = os.path.join(tmp_dir, input_key)
                connector_data = i['connector']
                internal = {URL_SCHEME_IDENTIFIER: path}

                if connector_class == 'File':
                    connector_manager.receive_cleanup(connector_data, input_key, internal)
                elif connector_class == 'Directory':
                    connector_manager.receive_directory_cleanup(connector_data, input_key, internal)

        elif isinstance(arg, dict):
            # connector_class should be one of 'File' or 'Directory'
            connector_class = arg['class']
            input_key = key
            path = os.path.join(tmp_dir, input_key)
            connector_data = val['connector']
            internal = {URL_SCHEME_IDENTIFIER: path}

            if connector_class == 'File':
                connector_manager.receive_cleanup(connector_data, input_key, internal)
            elif connector_class == 'Directory':
                connector_manager.receive_directory_cleanup(connector_data, input_key, internal)

    try:
        os.rmdir(tmp_dir)
    except (OSError, FileNotFoundError):
        # Maybe, raise a warning here, because not all connectors have cleaned up their contents correctly.
        pass