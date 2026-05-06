def read_long_description(readme_file):
    """ Read package long description from README file """
    try:
        import pypandoc
    except (ImportError, OSError) as exception:
        print('No pypandoc or pandoc: %s' % (exception,))
        if sys.version_info.major == 3:
            handle = open(readme_file, encoding='utf-8')
        else:
            handle = open(readme_file)
        long_description = handle.read()
        handle.close()
        return long_description
    else:
        return pypandoc.convert(readme_file, 'rst')