def putfile(self, filepath, buildroot, metahash):
        """Put a file in the cache.

        Args:
          filepath: Path to file on disk.
          buildroot: Path to buildroot
          buildrule: The rule that generated this file.
          metahash: hash object
        """
        def gen_obj_path(filename):
            filehash = util.hash_file(filepath).hexdigest()
            return filehash, os.path.join(self.obj_cachedir, filehash[0:2],
                                          filehash[2:4], filehash)

        filepath_relative = filepath.split(buildroot)[1][1:]  # Strip leading /
        # Path for the metahashed reference:
        incachepath = self._genpath(filepath_relative, metahash)

        filehash, obj_path = gen_obj_path(filepath)
        if not os.path.exists(obj_path):
            obj_dir = os.path.dirname(obj_path)
            if not os.path.exists(obj_dir):
                os.makedirs(obj_dir)
            log.debug('Adding to obj cache: %s -> %s', filepath, obj_path)
            os.link(filepath, obj_path)

        if os.path.exists(incachepath):
            existingfile_hash = util.hash_file(incachepath).hexdigest()
            if filehash != existingfile_hash:
                log.warn('File found in mh cache, but checksum differs. '
                         'Replacing with this new version. (File: %s)',
                         filepath)
                log.warn('Possible reasons for this:')
                log.warn(' 1. This build is not hermetic, and something '
                         'differs about the build environment compared to the '
                         'previous build.')
                log.warn(' 2. This file has a timestamp or other build-time '
                         'related data encoded into it, which will always '
                         'cause the checksum to differ when built.')
                log.warn(' 3. Everything is terrible and nothing works.')
                os.unlink(incachepath)

        if not os.path.exists(incachepath):
            log.debug('Adding to mh cache: %s -> %s', filepath, incachepath)
            if not os.path.exists(os.path.dirname(incachepath)):
                os.makedirs(os.path.dirname(incachepath))
            os.link(obj_path, incachepath)