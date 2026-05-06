def complete(self, filepath):
        '''
        Marks the item as complete by moving it to the done directory and optionally gzipping it.
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError("Can't Complete {}, it doesn't exist".format(filepath))
        if self._devel: self.logger.debug("Completing - {} ".format(filepath))
        if self.rotate_complete:
            try:
                complete_dir = str(self.rotate_complete())
            except Exception as e:
                self.logger.error("rotate_complete function failed with the following exception.")
                self.logger.exception(e)
                raise
            newdir = os.path.join(self._done_dir, complete_dir)
            newpath = os.path.join(newdir, os.path.split(filepath)[-1] )

            if not os.path.isdir(newdir):
                self.logger.debug("Making new directory: {}".format(newdir))
                os.makedirs(newdir)
        else:
            newpath = os.path.join(self._done_dir, os.path.split(filepath)[-1] )

        try:
            if self._compress_complete:
                if not filepath.endswith('.gz'):
                    #  Compressing complete, but existing file not compressed
                    #  Compress and move it and kick out
                    newpath += '.gz'
                    self._compress_and_move(filepath, newpath)
                    return newpath
                # else the file is already compressed and can just be moved
            #if not compressing completed file, just move it
            shutil.move(filepath, newpath)
            self.logger.info(" Completed - {}".format(filepath))
        except Exception as e:
            self.logger.error("Couldn't Complete {}".format(filepath))
            self.logger.exception(e)
            raise
        return newpath