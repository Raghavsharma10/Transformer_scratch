def process_tags(self, tag=None):
        """Process ID3 Tags for mp3 files."""
        if self.downloaded is False:
            raise serror("Track not downloaded, can't process tags..")
        filetype = magic.from_file(self.filepath, mime=True)
        if filetype != "audio/mpeg":
            raise serror("Cannot process tags for file type %s." % filetype)

        print("Processing tags for %s.." % self.filepath)
        if tag is None:
            tag = stag()
        tag.load_id3(self)
        tag.write_id3(self.filepath)