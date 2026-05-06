def parse_code(code, var_factory, **kwargs):
    """Parse a piece of text and substitude $var by either unique
    variable names or by the given kwargs mapping. Use $$ to escape $.

    Returns a CodeBlock and the resulting variable mapping.

    parse("$foo = $foo + $bar", bar="1")
    ("t1 = t1 + 1", {'foo': 't1', 'bar': '1'})
    """

    block = CodeBlock()
    defdict = collections.defaultdict(var_factory)
    defdict.update(kwargs)

    indent = -1
    code = code.strip()
    for line in code.splitlines():
        length = len(line)
        line = line.lstrip()
        spaces = length - len(line)
        if spaces:
            if indent < 0:
                indent = spaces
                level = 1
            else:
                level = spaces // indent
        else:
            level = 0

        # if there is a single variable and the to be inserted object
        # is a code block, insert the block with the current indentation level
        if line.startswith("$") and line.count("$") == 1:
            name = line[1:]
            if name in kwargs and isinstance(kwargs[name], CodeBlock):
                kwargs[name].write_into(block, level)
                continue

        block.write_line(string.Template(line).substitute(defdict), level)
    return block, dict(defdict)