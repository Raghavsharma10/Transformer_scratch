def main():
    """The entrypoint for the hairball command installed via setup.py."""
    description = ('PATH can be either the path to a scratch file, or a '
                   'directory containing scratch files. Multiple PATH '
                   'arguments can be provided.')
    parser = OptionParser(usage='%prog -p PLUGIN_NAME [options] PATH...',
                          description=description,
                          version='%prog {}'.format(__version__))
    parser.add_option('-d', '--plugin-dir', metavar='DIR',
                      help=('Specify the path to a directory containing '
                            'plugins. Plugins in this directory take '
                            'precedence over similarly named plugins '
                            'included with Hairball.'))
    parser.add_option('-p', '--plugin', action='append',
                      help=('Use the named plugin to perform analysis. '
                            'This option can be provided multiple times.'))
    parser.add_option('-k', '--kurt-plugin', action='append',
                      help=('Provide either a python import path (e.g, '
                            'kelp.octopi) to a package/module, or the path'
                            ' to a python file, which will be loaded as a '
                            'Kurt plugin. This option can be provided '
                            'multiple times.'))
    parser.add_option('-q', '--quiet', action='store_true',
                      help=('Prevent output from Hairball. Plugins may still '
                            'produce output.'))
    parser.add_option('-C', '--no-cache', action='store_true',
                      help='Do not use Hairball\'s cache.', default=False)
    options, args = parser.parse_args(sys.argv[1:])

    if not options.plugin:
        parser.error('At least one plugin must be specified via -p.')
    if not args:
        parser.error('At least one PATH must be provided.')

    if options.plugin_dir:
        if os.path.isdir(options.plugin_dir):
            sys.path.append(options.plugin_dir)
        else:
            parser.error('{} is not a directory'.format(options.plugin_dir))

    hairball = Hairball(options, args, cache=not options.no_cache)
    hairball.initialize_plugins()
    hairball.process()
    hairball.finalize()