def configure(default=None, dev=None):
    """
    The inner control loops for user interaction during quickstart
    configuration.
    """
    cache_loc = openaccess_epub.utils.cache_location()
    config_loc = openaccess_epub.utils.config_location()

    #Make the cache directory
    openaccess_epub.utils.mkdir_p(cache_loc)

    defaults = {'now': time.asctime(),
                'oae-version': openaccess_epub.__version__,
                'cache-location': unix_path_coercion(cache_loc),
                'input-relative-images': 'images-*',
                'use-input-relative-images': 'y',
                'image-cache': os.path.join(cache_loc, 'img_cache'),
                'use-image-cache': 'n',
                'use-image-fetching': 'y',
                'default-output': '.',
                'input-relative-css': '.',
                'epubcheck-jarfile': os.path.join(cache_loc,
                                                 'epubcheck-3.0',
                                                 'epubcheck-3.0.jar')}

    if default or dev:  # Skip interactive and apply defaults
        #Pass through the validation/modification steps
        if dev:  # The only current difference between dev and default
            defaults['use-image-cache'] = 'y'
        defaults['input-relative-images'] = list_opts(defaults['input-relative-images'])
        defaults['use-input-relative-images'] = boolean(defaults['use-input-relative-images'])
        defaults['image-cache'] = absolute_path(defaults['image-cache'])
        defaults['use-image-cache'] = boolean(defaults['use-image-cache'])
        defaults['use-image-fetching'] = boolean(defaults['use-image-fetching'])
        defaults['default-output'] = nonempty(defaults['default-output'])
        defaults['input-relative-css'] = nonempty(defaults['input-relative-css'])
        defaults['epubcheck-jarfile'] = absolute_path(defaults['epubcheck-jarfile'])
        config = config_formatter(CONFIG_TEXT, defaults)

        with open(config_loc, 'wb') as conf_out:
            conf_out.write(bytes(config, 'UTF-8'))
        print('The config file has been written to {0}'.format(config_loc))
        return

    config_dict = {'now': time.asctime(),
                   'oae-version': openaccess_epub.__version__,
                   'cache-location': unix_path_coercion(cache_loc)}

    print('''\nWelcome to the interactive configuration for OpenAccess_EPUB''')
    print('''
Please enter values for the following settings. To accept the default value
for the settings, shown in brackets, just push Enter.

-------------------------------------------------------------------------------\
''')
    print('''
OpenAccess_EPUB defines a default cache location for the storage of various
data (and the global config.py file), this location is:\n\n{0}
'''.format(cache_loc))

    input('Press Enter to start...')

    #Image Configuration
    print('''
 -- Configure Image Behavior --

When OpenAccess_EPUB is executed using the oaepub script, it can find the
images for the input articles using the following strategies (in order of
preference):

 Input-Relative: a path relative to the input file
 Cached Images: locate the images in a cache
 Fetched Online: attempts to download from the Internet (may fail)

We'll configure some values for each of these, and you\'ll also have the option
to turn them off.''')
    #Input-relative image details
    print('''
Where should OpenAccess_EPUB look for images relative to the input file?
A star "*" may be used as a wildcard to match the name of the input file.
Multiple path values may be specified if separated by commas.''')
    user_prompt(config_dict, 'input-relative-images', 'Input-relative images?:',
                default=defaults['input-relative-images'], validator=list_opts)
    print('''
Should OpenAccess_EPUB look for images relative to the input file by default?\
''')
    user_prompt(config_dict, 'use-input-relative-images',
                'Use input-relative images?: (Y/n)',
                default=defaults['use-input-relative-images'],
                validator=boolean)
    #Image cache details
    print('''
Where should OpenAccess_EPUB place the image cache?''')
    user_prompt(config_dict, 'image-cache', 'Image cache?:',
                default=defaults['image-cache'],
                validator=absolute_path)
    print('''
Should OpenAccess_EPUB use the image cache by default? This feature is intended
for developers and testers without local access to the image files and will
consume extra disk space for storage.''')
    user_prompt(config_dict, 'use-image-cache', 'Use image cache?: (y/N)',
                default=defaults['use-image-cache'],
                validator=boolean)
    #Image fetching online details
    print('''
Should OpenAccess_EPUB attempt to download the images from the Internet? This
is not supported for all publishers and not 100% guaranteed to succeed, you may
need to download them manually if this does not work.''')
    user_prompt(config_dict, 'use-image-fetching', 'Attempt image download?: (Y/n)',
                default=defaults['use-image-fetching'],
                validator=boolean)
    #Output configuration
    print('''
 -- Configure Output Behavior --

OpenAccess_EPUB produces ePub and log files as output. The following options
will determine what is done with these.

Where should OpenAccess_EPUB place the output ePub and log files? If you supply
a relative path, the output path will be relative to the input; if you supply
an absolute path, the output will always be placed there. The default behavior
is to place them in the same directory as the input.''')
    user_prompt(config_dict, 'default-output', 'Output path?:',
                default=defaults['default-output'],
                validator=nonempty)
    print('''
 -- Configure CSS Behavior --

ePub files use CSS for improved styling, and ePub-readers must support a basic
subset of CSS functions. OpenAccess_EPUB provides a default CSS file, but a
manual one may be supplied, relative to the input. Please define an
appropriate input-relative path.''')
    user_prompt(config_dict, 'input-relative-css', 'Input-relative CSS path?:',
                default=defaults['input-relative-css'],
                validator=nonempty)
    print('''
 -- Configure EpubCheck --

EpubCheck is a program written and maintained by the IDPF as a tool to validate
ePub. In order to use it, your system must have Java installed and it is
recommended to use the latest version. Downloads of this program are found here:

https://github.com/IDPF/epubcheck/releases

Once you have downloaded the zip file for the program, unzip the archive and
write a path to the .jar file here.''')
    user_prompt(config_dict, 'epubcheck-jarfile', 'Absolute path to epubcheck?:',
                default=defaults['epubcheck-jarfile'], validator=absolute_path)
    #Write the config.py file
    config = config_formatter(CONFIG_TEXT, config_dict)
    with open(config_loc, 'wb') as conf_out:
        conf_out.write(bytes(config, 'UTF-8'))
    print('''
Done configuring OpenAccess_EPUB!''')