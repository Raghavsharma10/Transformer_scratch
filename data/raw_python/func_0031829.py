def create_tar_archive(self):
        """ Create a tar archive of the main simulation outputs.
        """
        #file filter
        EXCLUDE_FILES = glob.glob(os.path.join(self.savefolder, 'cells'))
        EXCLUDE_FILES += glob.glob(os.path.join(self.savefolder,
                                                'populations', 'subsamples'))
        EXCLUDE_FILES += glob.glob(os.path.join(self.savefolder,
                                                'raw_nest_output'))

        def filter_function(tarinfo):
            print(tarinfo.name)
            if len([f for f in EXCLUDE_FILES if os.path.split(tarinfo.name)[-1]
                    in os.path.split(f)[-1]]) > 0 or \
               len([f for f in EXCLUDE_FILES if os.path.split(tarinfo.path)[-1]
                    in os.path.split(f)[-1]]) > 0:
                print('excluding %s' % tarinfo.name)

                return None
            else:
                return tarinfo

        if RANK == 0:
            print('creating archive %s' % (self.savefolder + '.tar'))
            #open file
            f = tarfile.open(self.savefolder + '.tar', 'w')
            #avoid adding files to repo as /scratch/$USER/hybrid_model/...
            arcname = os.path.split(self.savefolder)[-1]

            f.add(name=self.savefolder,
                  arcname=arcname,
                  filter=filter_function)
            f.close()

        #resync
        COMM.Barrier()