def write_template_file(source, target, content):
    """Write a new file from a given pystache template file and content"""

    # print(formatTemplateFile(source, content))
    print(target)
    data = format_template_file(source, content)
    with open(target, 'w') as f:
        for line in data:
            if type(line) != str:
                line = line.encode('utf-8')
            f.write(line)