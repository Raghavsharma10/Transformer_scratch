def download_results(self, savedir=None, raw=True, calib=False, index=None):
        """Download the previously found and stored Opus obsids.

        Parameters
        ==========
        savedir: str or pathlib.Path, optional
            If the database root folder as defined by the config.ini should not be used,
            provide a different savedir here. It will be handed to PathManager.
        """
        obsids = self.obsids if index is None else [self.obsids[index]]
        for obsid in obsids:
            pm = io.PathManager(obsid.img_id, savedir=savedir)
            pm.basepath.mkdir(exist_ok=True)
            to_download = []
            if raw is True:
                to_download.extend(obsid.raw_urls)
            if calib is True:
                to_download.extend(obsid.calib_urls)
            for url in to_download:
                basename = Path(url).name
                print("Downloading", basename)
                store_path = str(pm.basepath / basename)
                try:
                    urlretrieve(url, store_path)
                except Exception as e:
                    urlretrieve(url.replace("https", "http"), store_path)
            return str(pm.basepath)