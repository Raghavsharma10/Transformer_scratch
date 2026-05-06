def run(self):
        """ run all configured stages """

        self.sanity_check()

# TODO - check for devel
#        if not self.version:
#            raise Exception("no version")
# XXX check attr exist
        if not self.release_environment:
            raise Exception("no instance name")

        time_start = time.time()

        cwd = os.getcwd()

        who = getpass.getuser()
        self._make_outdirs()

        append_notices = ""
        if hasattr(self, 'opt_end'):
            append_notices = ". shortened push, only to %s stage" % self.opt_end
        if self.is_devel:
            append_notices += ". devel build"
        if hasattr(self, 'append_notices'):
            append_notices += self.append_notices

        line = "%s %s %s by %s%s" % (
            sys.argv[0], self.version, self.release_environment, who, append_notices)
        b = 'deploy begin %s' % line
        e = 'deploy done %s' % line

        if self.chatty:
            self.alact(b)

        ok = False
        stage_passed = None
        try:
            for stage in self.stages[self.stage_start:self.stage_end]:
                self.debug_msg("stage %s starting" % (stage,))
                getattr(self, stage)()
                self.chdir(cwd)
                stage_passed = stage
                self.debug_msg("stage %s complete" % (stage,))
            ok = True
        finally:
            if not ok:
                if self.chatty:
                    if not stage_passed:
                        self.alact(
                            'deploy failed %s. completed no stages' % line)
                    else:
                        self.alact('deploy failed %s. completed %s' %
                                   (line, stage_passed))
        self.status_msg('[OK]')
        if self.chatty:
            self.alact('%s in %0.3f sec' % (e, time.time() - time_start))

        return 0