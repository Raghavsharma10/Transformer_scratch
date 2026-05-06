def act(self):
        """
        Carries out the action associated with Verify button
        """
        tname = self.val.get()
        g = get_root(self).globals
        g.clog.info('Checking ' + tname + ' in simbad')
        try:
            ret = checkSimbad(g, tname)
            if len(ret) == 0:
                self.verify.config(bg=g.COL['stop'])
                g.clog.warn('No matches to "' + tname + '" found.')
                if tname not in self.failures:
                    self.failures.append(tname)
            elif len(ret) == 1:
                self.verify.config(bg=g.COL['start'])
                g.clog.info(tname + ' verified OK in simbad')
                g.clog.info('Primary simbad name = ' + ret[0]['Name'])
                if tname not in self.successes:
                    self.successes.append(tname)
            else:
                g.clog.warn('More than one match to "' + tname + '" found')
                self.verify.config(bg=g.COL['stop'])
                if tname not in self.failures:
                    self.failures.append(tname)
        except urllib.error.URLError:
            g.clog.warn('Simbad lookup timed out')
        except socket.timeout:
            g.clog.warn('Simbad lookup timed out')