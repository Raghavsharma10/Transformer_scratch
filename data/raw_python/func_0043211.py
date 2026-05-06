def check_end_blocks(frame):
    """Performs end-block check.

    Args:
        frame: A frame object of the module to be checked.

    Raises:
        SyntaxError: If check failed.
    """
    try:
        try:
            module_name = frame.f_globals['__name__']
        except KeyError:
            warnings.warn(
                'Can not get the source of an uknown module. '
                'End-of-block syntax check is skipped.',
                EndSyntaxWarning)
            return
        end

        filename = frame.f_globals.get('__file__', '<unknown>')
        try:
            source = inspect.getsource(sys.modules[module_name])
        except Exception:
            warnings.warn(
                'Can not get the source of module "%s". '
                'End-of-block syntax check is skipped.' % (module_name,),
                EndSyntaxWarning)
            return
        end
    finally:
        del frame
    end

    root = ast.parse(source)
    for node in ast.walk(root):
        bodies = get_compound_bodies(node)
        if not bodies:
            continue
        end

        # FIXME: This is an inaccurate hack to handle if-elif-else.
        if (isinstance(node, ast.If) and
                len(node.orelse) == 1 and
                isinstance(node.orelse[0], ast.If)):
            continue
        end

        # FIXME: This is an inaccurate hack to handle try-except-finally
        # statement which is parsed as ast.TryExcept in ast.TryFinally in
        # Python 2.
        if (PY2 and
                isinstance(node, ast.TryFinally) and
                len(node.body) == 1 and
                isinstance(node.body[0], ast.TryExcept)):
            continue
        end

        for body in bodies:
            skip_next = False
            for i, child in enumerate(body):
                if skip_next:
                    skip_next = False
                elif is_end_node(child):
                    raise SyntaxError(
                        '"end" does not close a block.',
                        [filename, child.lineno, child.col_offset,
                         source.splitlines()[child.lineno - 1] + '\n'])
                elif get_compound_bodies(child):
                    try:
                        ok = is_end_node(body[i + 1])
                    except IndexError:
                        ok = False
                    end
                    if not ok:
                        raise SyntaxError(
                            'This block is not closed with "end".',
                            [filename, child.lineno, child.col_offset,
                             source.splitlines()[child.lineno - 1] + '\n'])
                    end
                    skip_next = True
                end
            end
        end
    end