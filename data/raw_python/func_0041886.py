def get_output(src):
    """
    parse lines looking for commands
    """
    output = ''
    lines = open(src.path, 'rU').readlines()
    for line in lines:
        m = re.match(config.import_regex,line)
        if m:
            include_path = os.path.abspath(src.dir + '/' + m.group('script'));
            if include_path not in config.sources:
                script = Script(include_path)
                script.parents.append(src)
                config.sources[script.path] = script
            include_file = config.sources[include_path]
            #require statements dont include if the file has already been included
            if include_file not in config.stack or m.group('command') == 'import':
                config.stack.append(include_file)
                output += get_output(include_file)
        else:
            output += line
    return output