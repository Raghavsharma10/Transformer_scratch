def refresh(self):
        """Update list of files, if there are changes.
        
        Calls underlying list_rtn for the particular science instrument.
        Typically, these routines search in the pysat provided path,
        pysat_data_dir/platform/name/tag/,
        where pysat_data_dir is set by pysat.utils.set_data_dir(path=path).
        

        """

        output_str = '{platform} {name} {tag} {sat_id}'
        output_str = output_str.format(platform=self._sat.platform,
                                       name=self._sat.name, tag=self._sat.tag, 
                                       sat_id=self._sat.sat_id)
        output_str = " ".join(("pysat is searching for", output_str, "files."))
        output_str = " ".join(output_str.split())
        print (output_str)
        
        info = self._sat._list_rtn(tag=self._sat.tag, sat_id=self._sat.sat_id,
                                   data_path=self.data_path,
                                   format_str=self.file_format)

        if not info.empty:
            print('Found {ll:d} of them.'.format(ll=len(info)))
        else:
            estr = "Unable to find any files that match the supplied template. "
            estr += "If you have the necessary files please check pysat "
            estr += "settings and file locations (e.g. pysat.pysat_dir)."
            print(estr)
        info = self._remove_data_dir_path(info)
        self._attach_files(info)
        self._store()