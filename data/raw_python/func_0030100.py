def init_debug(self):
        """Initialize debugging features, such as a handler for USR2 to print a trace"""
        import signal

        def debug_trace(sig, frame):
            """Interrupt running process, and provide a python prompt for interactive
            debugging."""

            self.log('Trace signal received')
            self.log(''.join(traceback.format_stack(frame)))

        signal.signal(signal.SIGUSR2, debug_trace)