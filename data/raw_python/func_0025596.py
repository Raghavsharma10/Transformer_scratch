def generic_insert_with_folder(folder_name, file_name, template_name, args):
    """
    In general if we need to put a file on a folder, we use this method
    """
    # First we make sure views are a package instead a file
    if not os.path.isdir(
        os.path.join(
            args['django_application_folder'],
            folder_name
        )
    ):
        os.mkdir(os.path.join(args['django_application_folder'], folder_name))
        codecs.open(
            os.path.join(
                args['django_application_folder'],
                folder_name,
                '__init__.py'
            ),
            'w+'
        )

    view_file = create_or_open(
        os.path.join(
            folder_name,
            '{}.py'.format(file_name)
        ), 
        '', 
        args
    )
    # Load content from template
    render_template_with_args_in_file(
        view_file, 
        os.path.join(
            BASE_TEMPLATES_DIR, 
            template_name
        ),
        model_name=args['model_name'],
        model_prefix=args['model_prefix'],
        model_name_lower=args['model_name'].lower(),
        application_name=args['django_application_folder'].split("/")[-1]
    )
    view_file.close()