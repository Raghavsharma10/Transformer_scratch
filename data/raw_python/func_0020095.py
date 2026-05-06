def save_model(self):
        '''
        Saves all of the de-trending information to disk in an `npz` file
        and saves the DVS as a `pdf`.

        '''

        # Save the data
        log.info("Saving data to '%s.npz'..." % self.name)
        d = dict(self.__dict__)
        d.pop('_weights', None)
        d.pop('_A', None)
        d.pop('_B', None)
        d.pop('_f', None)
        d.pop('_mK', None)
        d.pop('K', None)
        d.pop('dvs', None)
        d.pop('clobber', None)
        d.pop('clobber_tpf', None)
        d.pop('_mission', None)
        d.pop('debug', None)
        d.pop('transit_model', None)
        d.pop('_transit_model', None)
        np.savez(os.path.join(self.dir, self.name + '.npz'), **d)

        # Save the DVS
        pdf = PdfPages(os.path.join(self.dir, self.name + '.pdf'))
        pdf.savefig(self.dvs.fig)
        pl.close(self.dvs.fig)
        d = pdf.infodict()
        d['Title'] = 'EVEREST: %s de-trending of %s %d' % (
            self.name, self._mission.IDSTRING, self.ID)
        d['Author'] = 'Rodrigo Luger'
        pdf.close()