def _run(self, keep_successfull):
        """Interpret the parsed 010 AST
        :returns: PfpDom

        """

        # example self._ast.show():
        #    FileAST:
        #      Decl: data, [], [], []
        #        TypeDecl: data, []
        #          Struct: DATA
        #            Decl: a, [], [], []
        #              TypeDecl: a, []
        #                IdentifierType: ['char']
        #            Decl: b, [], [], []
        #              TypeDecl: b, []
        #                IdentifierType: ['char']
        #            Decl: c, [], [], []
        #              TypeDecl: c, []
        #                IdentifierType: ['char']
        #            Decl: d, [], [], []
        #              TypeDecl: d, []
        #                IdentifierType: ['char']

        self._dlog("interpreting template")

        try:
            # it is important to pass the stream in as the stream
            # may change (e.g. compressed data)
            res = self._handle_node(self._ast, None, None, self._stream)
        except errors.InterpReturn as e:
            # TODO handle exit/return codes (e.g. return -1)
            res = self._root
        except errors.InterpExit as e:
            res = self._root
        except Exception as e:
            if keep_successfull:
                # return the root and set _pfp__error
                res = self._root
                res._pfp__error = e

            else:
                exc_type, exc_obj, traceback = sys.exc_info()
                more_info = "\nException at {}:{}".format(
                    self._orig_filename,
                    self._coord.line
                )
                six.reraise(
                    errors.PfpError,
                    errors.PfpError(exc_obj.__class__.__name__ + ": " + exc_obj.args[0] + more_info if len(exc_obj.args) > 0 else more_info),
                    traceback
                )

        # final drop-in after everything has executed
        if self._break_type != self.BREAK_NONE:
            self.debugger.cmdloop("execution finished")

        types = self.get_types()
        res._pfp__types = types

        return res