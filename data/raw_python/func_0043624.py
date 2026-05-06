def _get_source(self):
        " Get source from CVS or filepath. "
        source_dir = op.join(self.deploy_dir, 'source')
        for tp, cmd in settings.SRC_CLONE:
            if self.src.startswith(tp + '+'):
                program = which(tp)
                assert program, '%s not found.' % tp
                cmd = cmd % dict(src=self.src[len(tp) + 1:],
                                 source_dir=source_dir,
                                 branch=self.branch)
                cmd = "sudo -u %s %s" % (self['src_user'], cmd)
                call(cmd, shell=True)
                self.templates.append('src-%s' % tp)
                break
        else:
            self.templates.append('src-dir')
            copytree(self.src, source_dir)

        return source_dir