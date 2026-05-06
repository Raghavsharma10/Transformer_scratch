def do_eval(self, args):
        """Eval the user-supplied statement. Note that you can do anything with
        this command that you can do in a template.

        The resulting value of your statement will be displayed.
        """
        try:
            res = self._interp.eval(args)
            if res is not None:
                if hasattr(res, "_pfp__show"):
                    print(res._pfp__show())
                else:
                    print(repr(res))
        except errors.UnresolvedID as e:
            print("ERROR: " + e.message)
        except Exception as e:
            raise
            print("ERROR: " + e.message)
            
        return False