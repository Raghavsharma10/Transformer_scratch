def __find_caller(stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source file name,
        line number and function name.
        """
        frame = logging.currentframe()
        # On some versions of IronPython, currentframe() returns None if
        # IronPython isn't run with -X:Frames.
        if frame:
            frame = frame.f_back

        caller_info = '(unknown file)', 0, '(unknown function)', None

        while hasattr(frame, 'f_code'):
            co = frame.f_code
            if _logone_src in os.path.normcase(co.co_filename):
                frame = frame.f_back
                continue

            tb_info = None
            if stack_info:
                with StringIO() as _buffer:
                    _buffer.write('Traceback (most recent call last):\n')
                    traceback.print_stack(frame, file=_buffer)
                    tb_info = _buffer.getvalue().strip()

            caller_info = co.co_filename, frame.f_lineno, co.co_name, tb_info
            break
        return caller_info