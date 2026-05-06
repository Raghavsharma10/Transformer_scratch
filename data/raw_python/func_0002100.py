def download_previews(self, savedir=None):
        """Download preview files for the previously found and stored Opus obsids.

        Parameters
        ==========
        savedir: str or pathlib.Path, optional
            If the database root folder as defined by the config.ini should not be used,
            provide a different savedir here. It will be handed to PathManager.
        """
        for obsid in self.obsids:
            pm = io.PathManager(obsid.img_id, savedir=savedir)
            pm.basepath.mkdir(exist_ok=True)
            basename = Path(obsid.medium_img_url).name
            print("Downloading", basename)
            urlretrieve(obsid.medium_img_url, str(pm.basepath / basename))