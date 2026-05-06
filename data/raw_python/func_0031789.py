def collect_data(self):
        """
        Collect LFPs, CSDs and soma traces from each simulated population,
        and save to file.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        #collect some measurements resolved per file and save to file
        for measure in ['LFP', 'CSD']:
            if measure in self.savelist:
                self.collectSingleContribs(measure)


        #calculate lfp from all cell contribs
        lfp = self.calc_signal_sum(measure='LFP')

        #calculate CSD in every lamina
        if self.calculateCSD:
            csd = self.calc_signal_sum(measure='CSD')

        if RANK == 0 and self.POPULATION_SIZE > 0:
            #saving LFPs
            if 'LFP' in self.savelist:
                fname = os.path.join(self.populations_path,
                                     self.output_file.format(self.y,
                                                             'LFP')+'.h5')
                f = h5py.File(fname, 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=lfp, compression=4)
                f.close()
                del lfp
                assert(os.path.isfile(fname))
                print('save lfp ok')


            #saving CSDs
            if 'CSD' in self.savelist and self.calculateCSD:
                fname = os.path.join(self.populations_path,
                                     self.output_file.format(self.y,
                                                             'CSD')+'.h5')
                f = h5py.File(fname, 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=csd, compression=4)
                f.close()
                del csd
                assert(os.path.isfile(fname))
                print('save CSD ok')


            #save the somatic placements:
            pop_soma_pos = np.zeros((self.POPULATION_SIZE, 3))
            keys = ['xpos', 'ypos', 'zpos']
            for i in range(self.POPULATION_SIZE):
                for j in range(3):
                    pop_soma_pos[i, j] = self.pop_soma_pos[i][keys[j]]
            fname = os.path.join(self.populations_path,
                                 self.output_file.format(self.y, 'somapos.gdf'))
            np.savetxt(fname, pop_soma_pos)
            assert(os.path.isfile(fname))
            print('save somapos ok')

            #save rotations using hdf5
            fname = os.path.join(self.populations_path,
                                    self.output_file.format(self.y, 'rotations.h5'))
            f = h5py.File(fname, 'w')
            f.create_dataset('x', (len(self.rotations),))
            f.create_dataset('y', (len(self.rotations),))
            f.create_dataset('z', (len(self.rotations),))

            for i, rot in enumerate(self.rotations):
                for key, value in list(rot.items()):
                    f[key][i] = value
            f.close()
            assert(os.path.isfile(fname))
            print('save rotations ok')


        #resync threads
        COMM.Barrier()