def do_enable():
    """
    Uncomment any lines that start with #import in the .pth file
    """
    try:
        _lines = []
        with open(vext_pth, mode='r') as f:
            for line in f.readlines():
                if line.startswith('#') and line[1:].lstrip().startswith('import '):
                    _lines.append(line[1:].lstrip())
                else:
                    _lines.append(line)

        try:
            os.unlink('%s.tmp' % vext_pth)
        except:
            pass

        with open('%s.tmp' % vext_pth, mode='w+') as f:
            f.writelines(_lines)

        try:
            os.unlink('%s~' % vext_pth)
        except:
            pass

        os.rename(vext_pth, '%s~' % vext_pth)
        os.rename('%s.tmp' % vext_pth, vext_pth)
    except IOError as e:
        if e.errno == 2:
            # vext file doesn't exist, recreate it.
            create_pth()