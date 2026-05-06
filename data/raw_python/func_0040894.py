def parse(self, **global_args):
        """Entry point to parsing a BUILD file.

        Args:
          **global_args: Variables to include in the parsing environment.
        """

        if self.build_file not in ParseContext._parsed:
            # http://en.wikipedia.org/wiki/Abstract_syntax_tree
            # http://martinfowler.com/books/dsl.html
            butcher_context = {}
            for str_to_exec in self._strs_to_exec:
                ast = compile(str_to_exec, '<string>', 'exec')
                exec_function(ast, butcher_context)

            with ParseContext.activate(self):
                startdir = os.path.abspath(os.curdir)
                try:
                    os.chdir(self.build_file.path_on_disk)
                    if self.build_file not in ParseContext._parsed:
                        ParseContext._parsed.add(self.build_file)
                        eval_globals = copy.copy(butcher_context)
                        eval_globals.update(
                            {'ROOT_DIR': self.build_file.path_on_disk,
                             '__file__': 'bogus please fix this'})
                        eval_globals.update(global_args)
                        exec_function(self.build_file.code, eval_globals)
                finally:
                    os.chdir(startdir)