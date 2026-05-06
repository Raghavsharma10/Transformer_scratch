def sanity_check(args):
    """
    Verify if the work folder is a django app.
    A valid django app always must have a models.py file
    :return: None
    """
    if not os.path.isfile(
        os.path.join(
            args['django_application_folder'],
            'models.py'
        )
    ):
        print("django_application_folder is not a Django application folder")
        sys.exit(1)