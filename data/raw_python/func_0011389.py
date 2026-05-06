def do_show(self, args):
        """Show the current structure of __root (no args),
        or show the result of the expression (something that can be eval'd).
        """
        args = args.strip()

        to_show = self._interp._root
        if args != "":
            try:
                to_show = self._interp.eval(args)
            except Exception as e:
                print("ERROR: " + e.message)
                return False

        if hasattr(to_show, "_pfp__show"):
            print(to_show._pfp__show())
        else:
            print(repr(to_show))