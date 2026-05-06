def configure(self, pipe_config):
        """Configure from a dict"""

        # Create a context for evaluating the code for each pipeline. This removes the need
        # to qualify the class names with the module
        import ambry.etl
        import sys
        # ambry.build comes from ambry.bundle.files.PythonSourceFile#import_bundle
        eval_locals = dict(list(locals().items()) + list(ambry.etl.__dict__.items()) +
                           list(sys.modules['ambry.build'].__dict__.items()))

        replacements = {}

        def eval_pipe(pipe):
            if isinstance(pipe, string_types):
                try:
                    return eval(pipe, {}, eval_locals)
                except SyntaxError as e:
                    raise SyntaxError("SyntaxError while parsing pipe '{}' from metadata: {}"
                                      .format(pipe, e))
            else:
                return pipe

        def pipe_location(pipe):
            """Return a location prefix from a pipe, or None if there isn't one """
            if not isinstance(pipe, string_types):
                return None

            elif pipe[0] in '+-$!':
                return pipe[0]

            else:
                return None

        for segment_name, pipes in list(pipe_config.items()):
            if segment_name == 'final':
                # The 'final' segment is actually a list of names of Bundle methods to call afer the pipeline
                # completes
                super(Pipeline, self).__setattr__('final', pipes)
            elif segment_name == 'replace':
                for frm, to in iteritems(pipes):
                    self.replace(eval_pipe(frm), eval_pipe(to))
            else:

                # Check if any of the pipes have a location command. If not, the pipe
                # is cleared and the set of pipes replaces the ones that are there.
                if not any(bool(pipe_location(pipe)) for pipe in pipes):
                    # Nope, they are all clean
                    self[segment_name] = [eval_pipe(pipe) for pipe in pipes]
                else:
                    for i, pipe in enumerate(pipes):

                        if pipe_location(pipe):  # The pipe is prefixed with a location command
                            location = pipe_location(pipe)
                            pipe = pipe[1:]
                        else:
                            raise PipelineError(
                                'If any pipes in a section have a location command, they all must'
                                ' Segment: {} pipes: {}'.format(segment_name, pipes))

                        ep = eval_pipe(pipe)

                        if location == '+':  # append to the segment
                            self[segment_name].append(ep)
                        elif location == '-':  # Prepend to the segment
                            self[segment_name].prepend(ep)
                        elif location == '!':  # Replace a pipe of the same class

                            if isinstance(ep, type):
                                repl_class = ep
                            else:
                                repl_class = ep.__class__

                            self.replace(repl_class, ep, segment_name)