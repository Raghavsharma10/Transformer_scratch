def _get_emptydirs(self, files):
        '''Find empty directories and return them
        Only works for actual files in dap'''
        emptydirs = []
        for f in files:
            if self._is_dir(f):
                empty = True
                for ff in files:
                    if ff.startswith(f + '/'):
                        empty = False
                        break
                if empty:
                    emptydirs.append(f)
        return emptydirs