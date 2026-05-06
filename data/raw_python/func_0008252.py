def dumpJSON(self):
        """
        Encodes current parameters to JSON compatible dictionary
        """
        g = get_root(self).globals
        dtype = g.observe.rtype()
        if dtype == 'bias':
            target = 'BIAS'
        elif dtype == 'flat':
            target = 'FLAT'
        elif dtype == 'dark':
            target = 'DARK'
        else:
            target = self.target.value()

        return dict(
            target=target,
            ID=self.prog_ob.progid.value(),
            PI=self.pi.value(),
            OB='{:04d}'.format(self.prog_ob.obid.value()),
            Observers=self.observers.value(),
            comment=self.comment.value(),
            flags=dtype,
            filters=self.filter.value()
        )