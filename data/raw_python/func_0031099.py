def merge_dictionaries(current, new, only_defaults=False, template_special_case=False):
    '''
    Merge two settings dictionaries, recording how many changes were needed.

    '''
    changes = 0
    for key, value in new.items():
        if key not in current:
            if hasattr(global_settings, key):
                current[key] = getattr(global_settings, key)
                LOGGER.debug("Set %r to global default %r.", key, current[key])
            else:
                current[key] = copy.copy(value)
                LOGGER.debug("Set %r to %r.", key, current[key])
                changes += 1
                continue
        elif only_defaults:
            continue
        current_value = current[key]
        if hasattr(current_value, 'items'):
            changes += merge_dictionaries(current_value, value)
        elif isinstance(current_value, (list, tuple)):
            for element in value:
                if element not in current_value:
                    if template_special_case and key == 'TEMPLATES':
                        existing_matches = [
                            template for template in current_value if template['BACKEND'] == element['BACKEND']
                        ]
                        if existing_matches:
                            changes += merge_dictionaries(existing_matches[0], element)
                        else:
                            current[key] = list(current_value) + [element]
                            LOGGER.debug("Added %r to %r.", element, key)
                            changes += 1
                    else:
                        current[key] = list(current_value) + [element]
                        LOGGER.debug("Added %r to %r.", element, key)
                        changes += 1
        elif isinstance(current_value, Promise) or isinstance(value, Promise):
            # If we don't know what to do with it, replace it.
            if current_value is not value:
                current[key] = value
                LOGGER.debug("Set %r to %r.", key, current[key])
                changes += 1
        else:
            # If we don't know what to do with it, replace it.
            if current_value != value:
                current[key] = value
                LOGGER.debug("Set %r to %r.", key, current[key])
                changes += 1
    return changes