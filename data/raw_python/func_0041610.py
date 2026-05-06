def install_gem(gemname, version=None, conservative=True, ri=False, rdoc=False,
                development=False, format_executable=False, force=False,
                gem_source=None):
    """Install a ruby gem."""
    cmdline = ['gem', 'install']
    if conservative:
        cmdline.append('--conservative')
    if ri:
        cmdline.append('--ri')
    else:
        cmdline.append('--no-ri')
    if rdoc:
        cmdline.append('--rdoc')
    else:
        cmdline.append('--no-rdoc')
    if development:
        cmdline.append('--development')
    if format_executable:
        cmdline.append('--format-executable')
    if force:
        cmdline.append('--force')
    if version:
        cmdline.extend(['--version', version])
    cmdline.extend(['--clear-sources',
                    '--source', gem_source or RubyGems().gem_source])

    cmdline.append(gemname)

    msg = 'Installing ruby gem: %s' % gemname
    if version:
        msg += ' Version requested: %s' % version
    log.debug(msg)

    try:
        subprocess.check_output(cmdline, shell=False)
    except (OSError, subprocess.CalledProcessError) as err:
        raise error.ButcherError(
            'Gem install failed. Error was: %s. Output: %s' % (
                err, err.output))