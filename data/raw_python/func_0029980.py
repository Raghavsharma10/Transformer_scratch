def clean_build(self):
        """Delete the build directory and all ingested files """
        import shutil

        if self.build_fs.exists:
            try:
                shutil.rmtree(self.build_fs.getsyspath('/'))
            except NoSysPathError:
                pass