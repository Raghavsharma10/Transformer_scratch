def rock(self):
        """Starts and does the parsing."""
        if not self.argv:
            self.arg.view()
        while(self.argv):
            arg = self.argv.popleft()
            if arg == "-h" or arg == "--help":
                print(
                    """Usage: td [-h (--help)] [-v (--version)] [command]"""
                    """, where [command] is one of:\n\n"""
                    """v (view)\tChanges the way next output"""
                    """ will look like. See [td v -h].\n"""
                    """m (modify)\tApplies one time changes to"""
                    """ the database. See [td m -h].\n"""
                    """o (options)\tSets persistent options, applied"""
                    """ on every next execution. See [td o -h].\n"""
                    """a (add)\t\tAdds new item. See [td a -h].\n"""
                    """e (edit)\tEdits existing item. See [td e -h].\n"""
                    """r (rm)\t\tRemoves existing item. See [td r -h].\n"""
                    """d (done)\tMarks items as done. See [td d -h].\n"""
                    """D (undone)\tMarks items as not done. See [td D -h].\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\tShows this screen.\n"""
                    """  -v (--version)Shows version number."""
                )
            elif arg == "-v" or arg == "--version":
                print("td :: {}".format(__version__))
            elif arg == "v" or arg == "view":
                self._part("view", self.arg.view, {
                    "--no-color": ("nocolor", False),
                    "-s": ("sort", True), "--sort": ("sort", True),
                    "-p": ("purge", False), "--purge": ("purge", False),
                    "-d": ("done", True), "--done": ("done", True),
                    "-D": ("undone", True), "--undone": ("undone", True)
                },
                    """Usage: td v [-h (--help)] [command(s)]"""
                    """, where [command(s)] are any of:\n\n"""
                    """-s (--sort) <pattern>\tSorts the output using"""
                    """ <pattern>.\n"""
                    """-p (--purge)\t\tHides items marked as done.\n"""
                    """-d (--done) <pattern>\tDisplays items matching"""
                    """ <pattern> as done.\n"""
                    """-D (--undone) <pattern>\tDisplays items matching"""
                    """ <pattern> as not done.\n"""
                    """--no-color\t\tDo not add color codes to the output.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\t\tShows this screen."""
                )
            elif arg == "m" or arg == "modify":
                self._part("modify", self.arg.modify, {
                    "-s": ("sort", True), "--sort": ("sort", True),
                    "-p": ("purge", False), "--purge": ("purge", False),
                    "-d": ("done", True), "--done": ("done", True),
                    "-D": ("undone", True), "--undone": ("undone", True)
                },
                    """Usage: td m [-h (--help)] [command(s)]"""
                    """, where [command(s)] are any of:\n\n"""
                    """-s (--sort) <pattern>\tSorts database using"""
                    """ <pattern>.\n"""
                    """-p (--purge)\t\tRemoves items marked as done.\n"""
                    """-d (--done) <pattern>\tMarks items matching"""
                    """ <pattern> as done.\n"""
                    """-D (--undone) <pattern>\tMarks items matching"""
                    """ <pattern> as not done.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\t\tShows this screen."""
                )
            elif arg == "a" or arg == "add":
                args = dict()
                if self.argv and self.arg.model.exists(self.argv[0]):
                    args["parent"] = self.argv.popleft()
                self._part("add", self.arg.add, {
                    "-n": ("name", True), "--name": ("name", True),
                    "-p": ("priority", True), "--priority": ("priority", True),
                    "-c": ("comment", True), "--comment": ("comment", True)
                },
                    """Usage: td a [-h (--help)] [parent] [command(s)]"""
                    """, where [command(s)] are any of:\n\n"""
                    """-n (--name) <text>\t\tSets item's name.\n"""
                    """-p (--priority) <no|name>\tSets item's priority.\n"""
                    """-c (--comment) <text>\t\tSets item's comment.\n"""
                    """\nIf [parent] index is specified, new item will"""
                    """ become it's child.\n"""
                    """If any of the arguments is omitted,"""
                    """ this command will launch an interactive session"""
                    """ letting the user supply the rest of them.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\t\t\tShows this screen.""",
                    **args
                )
            elif arg == "e" or arg == "edit":
                if not self.argv:
                    raise NotEnoughArgumentsError("edit")
                args = dict()
                if self.argv[0] not in ["-h", "--help"]:
                    args["index"] = self.argv.popleft()
                self._part("edit", self.arg.edit, {
                    "--parent": ("parent", True),
                    "-n": ("name", True), "--name": ("name", True),
                    "-p": ("priority", True), "--priority": ("priority", True),
                    "-c": ("comment", True), "--comment": ("comment", True)
                },
                    """Usage: td e [-h (--help)] <index> [command(s)]"""
                    """, where [command(s)] are any of:\n\n"""
                    """--parent <index>\t\tChanges item's parent.\n"""
                    """-n (--name) <text>\t\tChanges item's name.\n"""
                    """-p (--priority) <no|name>\tChanges item's priority.\n"""
                    """-c (--comment) <text>\t\tChanges item's comment.\n"""
                    """\nIndex argument is required and has to point at"""
                    """ an existing item.\n"""
                    """If any of the arguments is omitted, it will launch"""
                    """ an interactive session letting the user supply the"""
                    """ rest of them.\n"""
                    """\nAdditions options:\n"""
                    """  -h (--help)\t\t\tShows this screen.""",
                    **args
                )
            elif arg == "r" or arg == "rm":
                args = dict()
                if not self.argv:
                    raise NotEnoughArgumentsError("rm")
                elif self.argv[0] not in ["-h", "--help"]:
                    args["index"] = self.argv.popleft()
                self._part("rm", self.arg.rm, {
                },
                    """Usage: td r [-h (--help)] <index>\n\n"""
                    """Index argument is required and has to point at"""
                    """ an existing item.\n"""
                    """\nAdditions options:\n"""
                    """  -h (--help)\tShows this screen.""",
                    **args
                )
            elif arg == "d" or arg == "done":
                args = dict()
                if not self.argv:
                    raise NotEnoughArgumentsError("done")
                elif self.argv[0] not in ["-h", "--help"]:
                    args["index"] = self.argv.popleft()
                self._part("done", self.arg.done, {
                },
                    """Usage: td d [-h (--help)] <index>\n\n"""
                    """Index argument is required and has to point at"""
                    """ an existing item.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\tShows this screen.""",
                    **args
                )
            elif arg == "D" or arg == "undone":
                args = dict()
                if not self.argv:
                    raise NotEnoughArgumentsError("undone")
                elif self.argv[0] not in ["-h", "--help"]:
                    args["index"] = self.argv.popleft()
                self._part("undone", self.arg.undone, {
                },
                    """Usage: td D [-h (--help)] <index>\n\n"""
                    """Index argument is required and has to point at"""
                    """ an existing item.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\tShows this screen.""",
                    **args
                )
            elif arg == "o" or arg == "options":
                self._part("options", self.arg.options, {
                    "-g": ("glob", False), "--global": ("glob", False),
                    "-s": ("sort", True), "--sort": ("sort", True),
                    "-p": ("purge", False), "--purge": ("purge", False),
                    "-d": ("done", True), "--done": ("done", True),
                    "-D": ("undone", True), "--undone": ("undone", True)
                },
                    """Usage: td o [-h (--help)] [command(s)]"""
                    """, where [command(s)] are any of:\n\n"""
                    """-g (--global)\t\tApply specified options to all"""
                    """ ToDo lists (store in ~/.tdrc).\n"""
                    """-s (--sort) <pattern>\tAlways sorts using"""
                    """ <pattern>.\n"""
                    """-p (--purge)\t\tAlways removes items marked"""
                    """as done.\n"""
                    """-d (--done) <pattern>\tAlways marks items maching"""
                    """ <pattern> as done.\n"""
                    """-D (--undone) <pattern>\tAlways marks items maching"""
                    """ <pattern> as not done.\n"""
                    """\nAdditional options:\n"""
                    """  -h (--help)\t\tShows this screen."""
                )
            else:
                raise UnrecognizedCommandError("td", arg)