def pyle_evaluate(expressions=None, modules=(), inplace=False, files=None, print_traceback=False):
    """The main method of pyle."""

    eval_globals = {}

    eval_globals.update(STANDARD_MODULES)

    for module_arg in modules or ():
        for module in module_arg.strip().split(","):
            module = module.strip()
            if module:
                eval_globals[module] = __import__(module)

    if not expressions:
        # Default 'do nothing' program
        expressions = ['line']

    files = files or ['-']
    eval_locals = {}
    for file in files:
        if file == '-':
            file = sys.stdin

        out_buf = sys.stdout if not inplace else StringIO.StringIO()

        out_line = None
        with (open(file, 'rb') if not hasattr(file, 'read') else file) as in_file:
            for num, line in enumerate(in_file.readlines()):
                was_whole_line = False
                if line[-1] == '\n':
                    was_whole_line = True
                    line = line[:-1]

                expr = ""
                try:
                    for expr in expressions:
                        words = [word.strip()
                                 for word in re.split(r'\s+', line)
                                 if word]
                        eval_locals.update({
                            'line': line, 'words': words,
                            'filename': in_file.name, 'num': num
                            })

                        out_line = eval(expr, eval_globals, eval_locals)

                        if out_line is None:
                            continue

                        # If the result is something list-like or iterable,
                        # output each item space separated.
                        if not isinstance(out_line, str) and not isinstance(out_line, unicode):
                            try:
                                out_line = u' '.join(unicode(part)
                                                     for part in out_line)
                            except:
                                # Guess it wasn't a list after all.
                                out_line = unicode(out_line)

                        line = out_line
                except Exception as e:
                    sys.stdout.flush()
                    sys.stderr.write("At %s:%d ('%s'): `%s`: %s\n" % (
                        in_file.name, num, truncate_ellipsis(line), expr, e))
                    if print_traceback:
                        traceback.print_exc(None, sys.stderr)
                else:
                    if out_line is None:
                        continue

                    out_line = out_line or u''
                    out_buf.write(out_line)
                    if was_whole_line:
                        out_buf.write('\n')
        if inplace:
            with open(file, 'wb') as out_file:
                out_file.write(out_buf.getvalue())
            out_buf.close()