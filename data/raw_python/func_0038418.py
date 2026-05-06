def convert(self):
        """Convert file in mp3 format."""
        if self.downloaded is False:
            raise serror("Track not downloaded, can't convert file..")
        filetype = magic.from_file(self.filepath, mime=True)
        if filetype == "audio/mpeg":
            print("File is already in mp3 format. Skipping convert.")
            return False

        rootpath = os.path.dirname(os.path.dirname(self.filepath))
        backupdir = rootpath + "/backups/" + self.get("username")
        if not os.path.exists(backupdir):
            os.makedirs(backupdir)

        backupfile = "%s/%s%s" % (
            backupdir,
            self.gen_filename(),
            self.get_file_extension(self.filepath))
        newfile = "%s.mp3" % self.filename_without_extension()

        os.rename(self.filepath, backupfile)
        self.filepath = newfile

        print("Converting to %s.." % newfile)
        song = AudioSegment.from_file(backupfile)
        return song.export(newfile, format="mp3")