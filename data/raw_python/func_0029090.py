def run(self):
        """Import the controller and run it.

        This mimics the processing done by :func:`helper.start`
        when a controller is run in the foreground.  A new instance
        of ``self.controller`` is created and run until a keyboard
        interrupt occurs or the controller stops on its own accord.

        """
        segments = self.controller.split('.')
        controller_class = reduce(getattr, segments[1:],
                                  __import__('.'.join(segments[:-1])))
        cmd_line = ['-f']
        if self.configuration is not None:
            cmd_line.extend(['-c', self.configuration])
        args = parser.get().parse_args(cmd_line)
        controller_instance = controller_class(args, platform)
        try:
            controller_instance.start()
        except KeyboardInterrupt:
            controller_instance.stop()