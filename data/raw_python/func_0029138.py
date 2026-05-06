def svg_output(dotfile, outfile='cloudformation.svg'):
    """Render template into svg file using the dot command (must be installed).

    :param dotfile: path to the dotfile
    :param outfile: filename for the output file
    :return:
    """
    try:
        cmd = ['dot', '-Tsvg', '-o' + outfile, dotfile]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            '\033[01;31mError running command: %s resulted in the ' % e.cmd +
            'following error: \033[01;32m %s' % e.output)
        return 1

    return 0