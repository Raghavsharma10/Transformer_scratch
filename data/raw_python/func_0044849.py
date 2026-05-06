def run(self, progress=True, verbose=False):
        """Compute all steps of the simulation. Be careful: if tmax is not set,
        this function will result in an infinit loop.

        Returns
        -------

        (t, fields):
            last time and result fields.
        """
        total_iter = int((self.tmax // self.user_dt) if self.tmax else None)
        log = logging.info if verbose else logging.debug
        if progress:
            with tqdm(initial=(self.i if self.i < total_iter else total_iter),
                      total=total_iter) as pbar:
                for t, fields in self:
                    pbar.update(1)
                    log("%s running: t: %g" % (self.id, t))
                try:
                    return t, fields
                except UnboundLocalError:
                    warnings.warn("Simulation already ended")
        for t, fields in self:
            log("%s running: t: %g" % (self.id, t))
        try:
            return t, fields
        except UnboundLocalError:
            warnings.warn("Simulation already ended")