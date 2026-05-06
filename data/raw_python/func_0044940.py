def djpl_compilemessages():
    """
    Compile messages hook for django_productline, this task checks for the activated languages
    in settings.LANGUAGES. It runs the standard django compilemessages management command with the -l parameter.
    Example language setting:
        LANGUAGES = [
            ('en', 'English'),
            ('de', 'Deutsch')
        ]
    Remarks:
        - Each argument for the management command MUST be a single list item, e.g. ['compilemessages', '--locale', 'en']
        - The compilemessages command MUST be executed in the projects root dir, so the CWD is adjusted before running this command.
    :return:
    """
    from django.conf import settings
    if hasattr(settings, 'LANGUAGES') and len(settings.LANGUAGES) > 0:
        # changing cwd to project root
        os.chdir(os.path.join(os.environ['APE_ROOT_DIR'], os.environ['CONTAINER_NAME']))
        languages = list()
        # collect the language abbreviations
        for language in settings.LANGUAGES:
            languages.append(language[0])
        args = ['compilemessages']
        # extend the arg list by the -locale argument for each language
        for lang in languages:
            args.extend(['--locale', str(lang)])
        tasks.manage(*args)