def get_obj(self, objpath, metahash, dst_path):
        """Get object from cache, write it to dst_path.

        Args:
          objpath: filename relative to buildroot
                   (example: mini-boot/blahblah/somefile.bin)
          metahash: metahash. See targets/base.py
          dst_path: Absolute path where the file should be written.
        Raises:
          CacheMiss: if the item is not in the cache
        """
        incachepath = self.path_in_cache(objpath, metahash)
        if not os.path.exists(incachepath):
            raise CacheMiss('%s not in cache.' % incachepath)
        else:
            log.debug('Cache hit! %s~%s', objpath, metahash.hexdigest())
            if not os.path.exists(os.path.dirname(dst_path)):
                os.makedirs(os.path.dirname(dst_path))
            os.link(incachepath, dst_path)