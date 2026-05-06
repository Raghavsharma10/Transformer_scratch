def main(argv):
    """
    Main function.
    """

    if len(argv) != 2:
        sys.stderr.write("\nYou can update a project in two steps.\n\n")
        sys.stderr.write("Step 1: Update or create infrastructure files\n")
        sys.stderr.write("        which will be needed to configure and build the project:\n")
        sys.stderr.write("        $ {0} --self\n\n".format(argv[0]))
        sys.stderr.write("Step 2: Create CMakeLists.txt and setup script in PROJECT_ROOT:\n")
        sys.stderr.write("        $ {0} <PROJECT_ROOT>\n".format(argv[0]))
        sys.stderr.write("        example:\n")
        sys.stderr.write("        $ {0} ..\n".format(argv[0]))
        sys.exit(-1)

    if argv[1] in ['-h', '--help']:
        print('Usage:')
        for t, h in [('python update.py --self',
                      'Update this script and fetch or update infrastructure files under autocmake/.'),
                     ('python update.py <builddir>',
                      '(Re)generate CMakeLists.txt and setup script and fetch or update CMake modules.'),
                     ('python update.py (-h | --help)',
                      'Show this help text.')]:
            print('  {0:30} {1}'.format(t, h))
        sys.exit(0)

    if argv[1] == '--self':
        # update self
        if not os.path.isfile('autocmake.yml'):
            print('- fetching example autocmake.yml')
            fetch_url(
                src='{0}example/autocmake.yml'.format(AUTOCMAKE_GITHUB_URL),
                dst='autocmake.yml'
            )
        if not os.path.isfile('.gitignore'):
            print('- creating .gitignore')
            with open('.gitignore', 'w') as f:
                f.write('*.pyc\n')
        for f in ['autocmake/configure.py',
                  'autocmake/__init__.py',
                  'autocmake/external/docopt.py',
                  'autocmake/external/__init__.py',
                  'autocmake/generate.py',
                  'autocmake/extract.py',
                  'autocmake/interpolate.py',
                  'autocmake/parse_rst.py',
                  'autocmake/parse_yaml.py',
                  'update.py']:
            print('- fetching {0}'.format(f))
            fetch_url(
                src='{0}{1}'.format(AUTOCMAKE_GITHUB_URL, f),
                dst='{0}'.format(f)
            )
        # finally create a README.md with licensing information
        with open('README.md', 'w') as f:
            print('- generating licensing information')
            f.write(licensing_info())
        sys.exit(0)

    process_yaml(argv)