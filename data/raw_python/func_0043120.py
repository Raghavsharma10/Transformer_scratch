def fake_exc_info(exc_info, filename, lineno):
    """Helper for `translate_exception`."""
    exc_type, exc_value, tb = exc_info

    # figure the real context out
    if tb is not None:
        # if there is a local called __tonnikala_exception__, we get
        # rid of it to not break the debug functionality.
        locals = tb.tb_frame.f_locals.copy()
        locals.pop('__tonnikala_exception__', None)
    else:
        locals = {}

    # assemble fake globals we need
    globals = {
        '__name__':                        filename,
        '__file__':                        filename,
        '__tonnikala_exception__':         exc_info[:2],

        # we don't want to keep the reference to the template around
        # to not cause circular dependencies, but we mark it as Tonnikala
        # frame for the ProcessedTraceback
        '__TK_template_info__':            None
    }

    # and fake the exception
    lineno = lineno or 0
    code = compile('\n' * (lineno - 1) + raise_helper, filename, 'exec')

    # if it's possible, change the name of the code.  This won't work
    # on some python environments such as google appengine
    try:
        if tb is None:
            location = 'template'
        else:
            function = tb.tb_frame.f_code.co_name
            if function == '__main__':
                location = 'top-level template code'
            elif function.startswith('__TK__block__'):
                location = 'block "%s"' % function[13:]
            elif function.startswith('__TK__typed__'):
                functype = function[13:].split('__')[0].replace('_', ' ')
                location = functype
            elif function.startswith('__TK_'):
                location = 'template'
            else:
                location = 'def "%s"' % function

        if not PY2:  # pragma: python3
            code = CodeType(0, code.co_kwonlyargcount, code.co_nlocals,
                            code.co_stacksize,
                            code.co_flags, code.co_code, code.co_consts,
                            code.co_names, code.co_varnames, filename,
                            location, code.co_firstlineno,
                            code.co_lnotab, (), ())

        else:  # pragma: python2
            code = CodeType(0, code.co_nlocals, code.co_stacksize,
                            code.co_flags, code.co_code, code.co_consts,
                            code.co_names, code.co_varnames, filename,
                            location, code.co_firstlineno,
                            code.co_lnotab, (), ())

    except Exception as e:
        pass

    # execute the code and catch the new traceback
    try:
        exec(code, globals, locals)
    except:
        exc_info = sys.exc_info()
        new_tb = exc_info[2].tb_next

    # return without this frame
    return exc_info[:2] + (new_tb,)