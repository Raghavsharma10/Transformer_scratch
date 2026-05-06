def get_objects(self, **kwargs):
        '''
        Run `rpm -q` command on a {local, remote} system to get back
        details of installed RPMs.

        Default rpm details extracted are as follows:
            * name
            * version
            * release
            * arch
            * nvra
            * license
            * os
            * packager
            * platform
            * sourcepackage
            * sourcerpm
            * summary
        '''
        fmt = ':::'.join('%%{%s}' % f for f in self._fields)
        if self.ssh_host:
            output = self._ssh_cmd(fmt)
        else:
            output = self._local_cmd(fmt)
        if isinstance(output, basestring):
            output = unicode(output, 'utf-8')
            output = output.strip().split('\n')
        lines = [l.strip().split(':::') for l in output]
        now = utcnow()
        host = self.ssh_host or socket.gethostname()
        for line in lines:
            obj = {'host': host, '_start': now}
            for i, item in enumerate(line):
                if item == '(none)':
                    item = None
                obj[self._fields[i]] = item
            obj['_oid'] = '%s__%s' % (host, obj['nvra'])
            self.objects.add(obj)
        return super(Rpm, self).get_objects(**kwargs)