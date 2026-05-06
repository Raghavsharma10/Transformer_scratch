def check(self, *args):
        """
        Checks the validity of the run parameters. Returns
        flag (True = OK), and a message which indicates the
        nature of the problem if the flag is False.
        """

        ok = True
        msg = ''
        g = get_root(self).globals
        dtype = g.observe.rtype()
        expert = g.cpars['expert_level'] > 0

        if dtype == 'bias' or dtype == 'flat' or dtype == 'dark':
            self.pi.configure(state='disable')
            self.prog_ob.configure(state='disable')
            self.target.disable()
        else:
            if expert:
                self.pi.configure(state='normal')
                self.prog_ob.configure(state='normal')
                self.prog_ob.enable()
            else:
                self.prog_ob.configure(state='disable')
                self.pi.configure(state='disable')
                self.prog_ob.disable()
            self.target.enable()

        if g.cpars['require_run_params']:
            if self.target.ok():
                self.target.entry.config(bg=g.COL['main'])
            else:
                self.target.entry.config(bg=g.COL['error'])
                ok = False
                msg += 'Target name field cannot be blank\n'

            if dtype == 'data caution' or \
               dtype == 'data' or dtype == 'technical':

                if self.prog_ob.ok():
                    self.prog_ob.config(bg=g.COL['main'])
                else:
                    self.prog_ob.config(bg=g.COL['error'])
                    ok = False
                    msg += 'Programme or OB ID field cannot be blank\n'

                if self.pi.ok():
                    self.pi.config(bg=g.COL['main'])
                else:
                    self.pi.config(bg=g.COL['error'])
                    ok = False
                    msg += 'Principal Investigator field cannot be blank\n'

            if self.observers.ok():
                self.observers.config(bg=g.COL['main'])
            else:
                self.observers.config(bg=g.COL['error'])
                ok = False
                msg += 'Observers field cannot be blank'
        return (ok, msg)