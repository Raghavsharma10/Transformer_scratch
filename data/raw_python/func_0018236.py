def cmdHELP(self, params):
        """[<command>]
        Display help
        Display either brief help on all commands, or detailed
        help on a single command passed as a parameter.
        """
        if params:
            cmd = params[0].upper()
            if self.COMMANDS.has_key(cmd):
                method = self.COMMANDS[cmd]
                doc = method.__doc__.split("\n")
                docp = doc[0].strip()
                docl = '\n'.join( [l.strip() for l in doc[2:]] )
                if not docl.strip():  # If there isn't anything here, use line 1
                    docl = doc[1].strip()
                self.writeline(
                    "%s %s\n\n%s" % (
                        cmd,
                        docp,
                        docl,
                    )
                )
                return
            else:
                self.writeline("Command '%s' not known" % cmd)
        else:
            self.writeline("Help on built in commands\n")
        keys = self.COMMANDS.keys()
        keys.sort()
        for cmd in keys:
            method = self.COMMANDS[cmd]
            if getattr(method, 'hidden', False):
                continue
            if method.__doc__ == None:
                self.writeline("no help for command %s" % method)
                return
            doc = method.__doc__.split("\n")
            docp = doc[0].strip()
            docs = doc[1].strip()
            if len(docp) > 0:
                docps = "%s - %s" % (docp, docs, )
            else:
                docps = "- %s" % (docs, )
            self.writeline(
                "%s %s" % (
                    cmd,
                    docps,
                )
            )