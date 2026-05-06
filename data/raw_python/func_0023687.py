def get_curline():
    """Return the current python source line."""
    if Frame:
        frame = Frame.get_selected_python_frame()
        if frame:
            line = ''
            f = frame.get_pyop()
            if f and not f.is_optimized_out():
                cwd = os.path.join(os.getcwd(), '')
                fname = f.filename()
                if cwd in fname:
                    fname = fname[len(cwd):]
                try:
                    line = f.current_line()
                except IOError:
                    pass
                if line:
                    # Use repr(line) to avoid UnicodeDecodeError on the
                    # following print invocation.
                    line = repr(line).strip("'")
                    line = line[:-2] if line.endswith(r'\n') else line
                    return ('-> %s(%s): %s' % (fname,
                                        f.current_line_num(), line))
    return ''