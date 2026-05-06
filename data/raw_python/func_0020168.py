def _save_npz(self):
        '''
        Saves all of the de-trending information to disk in an `npz` file

        '''

        # Save the data
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
        np.savez(os.path.join(self.dir, self.name + '.npz'), **d)