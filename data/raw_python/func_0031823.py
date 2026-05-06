def run(self):
        """ Perform the postprocessing steps, computing compound signals from
        cell-specific output files.
        """
        if RANK == 0:
            if 'LFP' in self.savelist:
                #get the per population LFPs and total LFP from all populations:
                self.LFPdict, self.LFPsum = self.calc_lfp()
                self.LFPdictLayer = self.calc_lfp_layer()
    
                #save global LFP sum, and from L23E, L4I etc.:
                f = h5py.File(os.path.join(self.savefolder,
                                           self.compound_file.format('LFP')
                                           ), 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=self.LFPsum, compression=4)
                f.close()
    
                for key, value in list(self.LFPdictLayer.items()):
                    f = h5py.File(os.path.join(self.populations_path,
                                               self.output_file.format(key,
                                                                       'LFP.h5')
                                               ), 'w')
                    f['srate'] = 1E3 / self.dt_output
                    f.create_dataset('data', data=value, compression=4)
                    f.close()

            if 'CSD' in self.savelist:
                #get the per population CSDs and total CSD from all populations:
                self.CSDdict, self.CSDsum = self.calc_csd()
                self.CSDdictLayer = self.calc_csd_layer()
    
                #save global CSD sum, and from L23E, L4I etc.:
                f = h5py.File(os.path.join(self.savefolder,
                                           self.compound_file.format('CSD')),
                              'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=self.CSDsum, compression=4)
                f.close()
    
                for key, value in list(self.CSDdictLayer.items()):
                    f = h5py.File(os.path.join(self.populations_path,
                                               self.output_file.format(key,
                                                                       'CSD.h5')
                                               ), 'w')
                    f['srate'] = 1E3 / self.dt_output
                    f.create_dataset('data', data=value, compression=4)
                    f.close()

        else:
            pass