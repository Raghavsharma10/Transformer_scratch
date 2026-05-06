def download_artwork(self, localdir, max_retry):
        """
        Download track's artwork and return file path.
        Artwork's path is saved in track's metadata as 'artwork-path' key.
        """
        if self.get("artwork-url") == "None":
            self.metadata["artwork-path"] = None
            return None

        artwork_dir = localdir + "/artworks"
        if not os.path.isdir(artwork_dir):
            if os.path.isfile(artwork_dir):
                os.unlink(artwork_dir)
            os.mkdir(artwork_dir)

        artwork_filepath = artwork_dir + "/" + self.gen_artwork_filename()

        retry = max_retry
        while True:
            try:
                res = urllib.request.urlopen(self.get("artwork-url"))
                with open(artwork_filepath, "wb") as file:
                    file.write(res.read())
                break
            except Exception as e:
                retry -= 1
                if retry < 0:
                    print(serror("Can't download track's artwork, max retry "
                                 "reached (%d). Error occured: %s" % (
                                     max_retry, type(e))))
                    return False
                else:
                    print("\033[93mTrack's artwork download failed (%s). "
                          "Retrying.. (%d/%d) \033[0m" % (
                              type(e),
                              max_retry - retry,
                              max_retry))

        self.metadata["artwork-path"] = artwork_filepath