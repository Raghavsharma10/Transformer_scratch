def merge_includes(code):
    """Merge all includes recursively."""

    pattern = '\#\s*include\s*"(?P<filename>[a-zA-Z0-9\_\-\.\/]+)"'
    regex = re.compile(pattern)
    includes = []

    def replace(match):
        filename = match.group("filename")

        if filename not in includes:
            includes.append(filename)
            path = glsl.find(filename)
            if not path:
                logger.critical('"%s" not found' % filename)
                raise RuntimeError("File not found", filename)
            text = '\n// --- start of "%s" ---\n' % filename
            with open(path) as fh:
                text += fh.read()
            text += '// --- end of "%s" ---\n' % filename
            return text
        return ''

    # Limit recursion to depth 10
    for i in range(10):
        if re.search(regex, code):
            code = re.sub(regex, replace, code)
        else:
            break

    return code