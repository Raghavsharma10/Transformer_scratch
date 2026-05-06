def user_exception(self, frame, exc_info):
        """This function is called if an exception occurs,
        but only if we are to stop at or just below this level."""
        exc_type, exc_value, exc_traceback = exc_info
        frame.f_locals['__exception__'] = exc_type, exc_value

        # An 'Internal StopIteration' exception is an exception debug event
        # issued by the interpreter when handling a subgenerator run with
        # 'yield from' or a generator controled by a for loop. No exception has
        # actually occured in this case. The debugger uses this debug event to
        # stop when the debuggee is returning from such generators.
        prefix = 'Internal ' if (PY34 and not exc_traceback
                                    and exc_type is StopIteration) else ''
        self.message('--Exception--\n%s%s' % (prefix,
            traceback.format_exception_only(exc_type, exc_value)[-1].strip()))
        self.interaction(frame, exc_traceback)