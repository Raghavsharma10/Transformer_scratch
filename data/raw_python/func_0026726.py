def _get_save_path(self, bury=False):
        """Return the path that this Entry should be saved to."""
        filename = self.get_filename(self[self._KEYS.NAME])

        # Put objects that shouldn't belong in this catalog in the boneyard
        if bury:
            outdir = self.catalog.get_repo_boneyard()

        # Get normal repository save directory
        else:
            repo_folders = self.catalog.PATHS.get_repo_output_folders()
            # If no repo folders exist, raise an error -- cannot save
            if not len(repo_folders):
                err_str = (
                    "No output data repositories found. Cannot save.\n"
                    "Make sure that repo names are correctly configured "
                    "in the `input/repos.json` file, and either manually or "
                    "automatically (using `astrocats CATALOG git-clone`) "
                    "clone the appropriate data repositories.")
                self.catalog.log.error(err_str)
                raise RuntimeError(err_str)

            outdir = repo_folders[0]

        return outdir, filename