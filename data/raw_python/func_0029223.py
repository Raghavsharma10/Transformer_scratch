def walk(self):
        """ Walk the directory like os.path
        (yields a 3-tuple (dirpath, dirnames, filenames)
        except it exclude all files/directories on the fly. """
        for root, dirs, files in os.walk(self.path, topdown=True):
            # TODO relative walk, recursive call if root excluder found???
            #root_excluder = get_root_excluder(root)
            ndirs = []
            # First we exclude directories
            for d in list(dirs):
                if self.is_excluded(os.path.join(root, d)):
                    dirs.remove(d)
                elif not os.path.islink(os.path.join(root, d)):
                    ndirs.append(d)

            nfiles = []
            for fpath in (os.path.join(root, f) for f in files):
                if not self.is_excluded(fpath) and not os.path.islink(fpath):
                    nfiles.append(os.path.relpath(fpath, root))

            yield root, ndirs, nfiles