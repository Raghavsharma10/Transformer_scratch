def clean_workspace(self):
        '''Clean up the temporary workspace if one exists'''
        if os.path.isdir(self._temp_workspace):
            shutil.rmtree(self._temp_workspace)