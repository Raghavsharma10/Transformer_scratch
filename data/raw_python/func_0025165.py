def convert_readme_to_rst():
    """ Attempt to convert a README.md file into README.rst """
    project_files = os.listdir('.')
    for filename in project_files:
        if filename.lower() == 'readme':
            raise ProjectError(
                'found {} in project directory...'.format(filename) +
                'not sure what to do with it, refusing to convert'
            )
        elif filename.lower() == 'readme.rst':
            raise ProjectError(
                'found {} in project directory...'.format(filename) +
                'refusing to overwrite'
            )
    for filename in project_files:
        if filename.lower() == 'readme.md':
            rst_filename = 'README.rst'
            logger.info('converting {} to {}'.format(filename, rst_filename))
            try:
                rst_content = pypandoc.convert(filename, 'rst')
                with open('README.rst', 'w') as rst_file:
                    rst_file.write(rst_content)
                return
            except OSError as e:
                raise ProjectError(
                    'could not convert readme to rst due to pypandoc error:' + os.linesep + str(e)
                )
    raise ProjectError('could not find any README.md file to convert')