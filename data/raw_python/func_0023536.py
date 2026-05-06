def do_break(self, arg, temporary = 0):
        """b(reak) [ ([filename:]lineno | function) [, condition] ]
        Without argument, list all breaks.

        With a line number argument, set a break at this line in the
        current file.  With a function name, set a break at the first
        executable line of that function.  If a second argument is
        present, it is a string specifying an expression which must
        evaluate to true before the breakpoint is honored.

        The line number may be prefixed with a filename and a colon,
        to specify a breakpoint in another file (probably one that
        hasn't been loaded yet).  The file is searched for on
        sys.path; the .py suffix may be omitted.
        """
        if not arg:
            all_breaks = '\n'.join(bp.bpformat() for bp in
                                bdb.Breakpoint.bpbynumber if bp)
            if all_breaks:
                self.message("Num Type         Disp Enb   Where")
                self.message(all_breaks)
            return

        # Parse arguments, comma has lowest precedence and cannot occur in
        # filename.
        args = arg.rsplit(',', 1)
        cond =  args[1].strip() if len(args) == 2 else None
        # Parse stuff before comma: [filename:]lineno | function.
        args = args[0].rsplit(':', 1)
        name = args[0].strip()
        lineno =  args[1] if len(args) == 2 else args[0]
        try:
            lineno = int(lineno)
        except ValueError:
            if len(args) == 2:
                self.error('Bad lineno: "{}".'.format(lineno))
            else:
                # Attempt the list of possible function or method fully
                # qualified names and corresponding filenames.
                candidates = get_fqn_fname(name, self.curframe)
                for fqn, fname in candidates:
                    try:
                        bp = self.set_break(fname, None, temporary, cond, fqn)
                        self.message('Breakpoint {:d} at {}:{:d}'.format(
                                                bp.number, bp.file, bp.line))
                        return
                    except bdb.BdbError:
                        pass
                if not candidates:
                    self.error(
                        'Not a function or a built-in: "{}"'.format(name))
                else:
                    self.error('Bad name: "{}".'.format(name))
        else:
            filename = self.curframe.f_code.co_filename
            if len(args) == 2 and name:
                filename = name
            if filename.startswith('<') and filename.endswith('>'):
                # allow <doctest name>: doctest installs a hook at
                # linecache.getlines to allow <doctest name> to be
                # linecached and readable.
                if filename == '<string>' and self.mainpyfile:
                    filename = self.mainpyfile
            else:
                root, ext = os.path.splitext(filename)
                if ext == '':
                    filename = filename + '.py'
                if not os.path.exists(filename):
                    self.error('Bad filename: "{}".'.format(arg))
                    return
            try:
                bp = self.set_break(filename, lineno, temporary, cond)
            except bdb.BdbError as err:
                self.error(err)
            else:
                self.message('Breakpoint {:d} at {}:{:d}'.format(
                                        bp.number, bp.file, bp.line))