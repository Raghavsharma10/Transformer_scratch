def create_or_open(file_name, initial_template_file_name, args):
    """
    Creates a file or open the file with file_name name
    :param file_name: String with a filename
    :param initial_template_file_name: String with path to initial template
    :param args: from console to determine path to save the files
    """
    file = None
    if not os.path.isfile(
        os.path.join(
            args['django_application_folder'],
            file_name
        )
    ):
        # If file_name does not exists, create
        file = codecs.open(
            os.path.join(
                args['django_application_folder'],
                file_name
            ),
            'w+',
            encoding='UTF-8'
        )
        print("Creating {}".format(file_name))
        if initial_template_file_name:
            render_template_with_args_in_file(file, initial_template_file_name, **{})
    else:
        # If file exists, just load the file
        file = codecs.open(
            os.path.join(
                args['django_application_folder'],
                file_name
            ),
            'a+',
            encoding='UTF-8'
        )

    return file