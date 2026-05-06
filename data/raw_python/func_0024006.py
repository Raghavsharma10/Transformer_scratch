def upload_path(instance, filename):
    '''
    This method is created to return the path to upload files. This path must be
    different from any other to avoid problems.
    '''
    path_separator = "/"
    date_separator = "-"
    ext_separator = "."
    empty_string = ""
    # get the model name
    model_name = model_inspect(instance)['modelname']

    # get the string date
    date = datetime.now().strftime("%Y-%m-%d").split(date_separator)
    curr_day = date[2]
    curr_month = date[1]
    curr_year = date[0]

    split_filename = filename.split(ext_separator)
    filename = empty_string.join(split_filename[:-1])
    file_ext = split_filename[-1]

    new_filename = empty_string.join([filename, str(random.random()).split(ext_separator)[1]])
    new_filename = ext_separator.join([new_filename, file_ext])
    string_path = path_separator.join([model_name, curr_year, curr_month, curr_day, new_filename])
    # the path is built using the current date and the modelname
    return string_path