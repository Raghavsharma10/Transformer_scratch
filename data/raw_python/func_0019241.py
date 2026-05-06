def move_dll(self):
        """Try to find the resulting dll file and to move it into the
        `cythons` package.

        Things to be aware of:
          * The file extension either `pyd` (Window) or `so` (Linux).
          * The folder containing the dll file is system dependent, but is
            always a subfolder of the `cythons` package.
          * Under Linux, the filename might contain system information, e.g.
            ...cpython-36m-x86_64-linux-gnu.so.
        """
        dirinfos = os.walk(self.buildpath)
        next(dirinfos)
        system_dependent_filename = None
        for dirinfo in dirinfos:
            for filename in dirinfo[2]:
                if (filename.startswith(self.cyname) and
                        filename.endswith(dllextension)):
                    system_dependent_filename = filename
                    break
            if system_dependent_filename:
                try:
                    shutil.move(os.path.join(dirinfo[0],
                                             system_dependent_filename),
                                os.path.join(self.cydirpath,
                                             self.cyname+dllextension))
                    break
                except BaseException:
                    prefix = ('After trying to cythonize module %s, when '
                              'trying to move the final cython module %s '
                              'from directory %s to directory %s'
                              % (self.pyname, system_dependent_filename,
                                 self.buildpath, self.cydirpath))
                    suffix = ('A likely error cause is that the cython module '
                              '%s does already exist in this directory and is '
                              'currently blocked by another Python process.  '
                              'Maybe it helps to close all Python processes '
                              'and restart the cyhonization afterwards.'
                              % self.cyname+dllextension)
                    objecttools.augment_excmessage(prefix, suffix)
        else:
            raise IOError('After trying to cythonize module %s, the resulting '
                          'file %s could neither be found in directory %s nor '
                          'its subdirectories.  The distul report should tell '
                          'whether the file has been stored somewhere else,'
                          'is named somehow else, or could not be build at '
                          'all.' % self.buildpath)