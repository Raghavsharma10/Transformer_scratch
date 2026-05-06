def red_get_mount_connectors(red_data, ignore_outputs):
    """
    Returns a list of mounting connectors

    :param red_data: The red data to be searched
    :param ignore_outputs: If outputs should be ignored
    :return: A list of connectors with active mount option.
    """

    keys = []

    batches = red_data.get('batches')
    inputs = red_data.get('inputs')
    if batches:
        for batch in batches:
            keys.extend(red_get_mount_connectors_from_inputs(batch['inputs']))
    elif inputs:
        keys.extend(red_get_mount_connectors_from_inputs(inputs))

    if not ignore_outputs:
        outputs = red_data.get('outputs')
        if batches:
            for batch in batches:
                batch_outputs = batch.get('outputs')
                if batch_outputs:
                    keys.extend(red_get_mount_connectors_from_outputs(batch_outputs))

        elif outputs:
            keys.extend(red_get_mount_connectors_from_outputs(outputs))

    return keys