def check_removable(dev, opts):
    ''' Removable drives can be identified under /sys. '''
    try:  # get parent device from sys filesystem, look from right.  :-/
        parent = os.readlink(f'/sys/class/block/{dev}').rsplit("/", 2)[1]
        with open(f'/sys/block/{parent}/removable') as f:
            return f.read() == '1\n'

    except IndexError as err:
        if opts.debug:
            print('ERROR: parent block device not found.', err)
    except IOError as err:
        if opts.debug:
            print('ERROR:', err)