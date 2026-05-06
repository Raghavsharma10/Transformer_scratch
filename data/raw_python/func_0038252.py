def gen_cmake_command(config):
    """
    Generate CMake command.
    """
    from autocmake.extract import extract_list

    s = []
    s.append("\n\ndef gen_cmake_command(options, arguments):")
    s.append('    """')
    s.append("    Generate CMake command based on options and arguments.")
    s.append('    """')
    s.append("    command = []")

    for env in config['export']:
        s.append('    command.append({0})'.format(env))

    s.append("    command.append(arguments['--cmake-executable'])")

    for definition in config['define']:
        s.append('    command.append({0})'.format(definition))

    s.append("    command.append('-DCMAKE_BUILD_TYPE={0}'.format(arguments['--type']))")
    s.append("    command.append('-G\"{0}\"'.format(arguments['--generator']))")
    s.append("    if arguments['--cmake-options'] != \"''\":")
    s.append("        command.append(arguments['--cmake-options'])")
    s.append("    if arguments['--prefix']:")
    s.append("        command.append('-DCMAKE_INSTALL_PREFIX=\"{0}\"'.format(arguments['--prefix']))")

    s.append("\n    return ' '.join(command)")

    return '\n'.join(s)