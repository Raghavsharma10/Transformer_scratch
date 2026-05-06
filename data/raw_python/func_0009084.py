def modify_level(level,field,values,append=True):
    '''modify level is intended to add / modify a content type.
    Default content type is list, meaning the entry is appended.
    If you set append to False, the content will be overwritten
    For any other content type, the entry is overwritten.
    '''
    field = field.lower()
    valid_fields = ['regexp','skip_files','include_files']
    if field not in valid_fields:
        bot.warning("%s is not a valid field, skipping. Choices are %s" %(field,",".join(valid_fields)))
        return level
    if append:
        if not isinstance(values,list):
            values = [values]
        if field in level:
            level[field] = level[field] + values
        else:
            level[field] = values
    else:
        level[field] = values

    level = make_level_set(level)

    return level