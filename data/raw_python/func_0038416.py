def download(self, localdir, max_retry):
        """ Download a track in local directory. """
        local_file = self.gen_localdir(localdir) + self.gen_filename()

        if self.track_exists(localdir):
            print("Track {0} already downloaded, skipping!".format(
                self.get("id")))
            return False

        if local_file in self.get_ignored_tracks(localdir):
            print("\033[93mTrack {0} ignored, skipping!!\033[0m".format(
                self.get("id")))
            return False

        dlurl = self.get_download_link()

        if not dlurl:
            raise serror("Can't download track_id:%d|%s" % (
                self.get("id"),
                self.get("title")))

        retry = max_retry
        print("\nDownloading %s (%d).." % (self.get("title"), self.get("id")))

        while True:
            try:
                urllib.request.urlretrieve(dlurl, local_file,
                                           self._progress_hook)
                break
            except Exception as e:
                if os.path.isfile(local_file):
                    os.unlink(local_file)
                retry -= 1

                if retry < 0:
                    raise serror("Can't download track-id %s, max retry "
                                 "reached (%d). Error occured: %s" % (
                                     self.get("id"), max_retry, type(e)))
                else:
                    print("\033[93mError occured for track-id %s (%s). "
                          "Retrying.. (%d/%d) \033[0m" % (
                              self.get("id"),
                              type(e),
                              max_retry - retry,
                              max_retry))
            except KeyboardInterrupt:
                if os.path.isfile(local_file):
                    os.unlink(local_file)
                raise serror("KeyBoard Interrupt: Incomplete file removed.")

        self.filepath = local_file + self.get_file_extension(local_file)
        os.rename(local_file, self.filepath)
        print("Downloaded => %s" % self.filepath)

        self.downloaded = True
        return True