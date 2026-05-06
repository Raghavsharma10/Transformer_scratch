def replace(self, repl_class, replacement, target_segment_name=None):
        """Replace a pipe segment, specified by its class, with another segment"""

        for segment_name, pipes in iteritems(self):

            if target_segment_name and segment_name != target_segment_name:
                raise Exception()

            repl_pipes = []
            found = False
            for pipe in pipes:
                if isinstance(pipe, repl_class):
                    pipe = replacement
                    found = True

                repl_pipes.append(pipe)

            if found:
                found = False
                self[segment_name] = repl_pipes