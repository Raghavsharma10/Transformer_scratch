def get_file_extension(self, filepath):
        """
        This method check mimetype to define file extension.
        If it can't, it use original-format metadata.
        """
        mtype = magic.from_file(filepath, mime=True)
        if type(mtype) == bytes:
            mtype = mtype.decode("utf-8")

        if mtype == "audio/mpeg":
            ext = ".mp3"
        elif mtype == "audio/x-wav":
            ext = ".wav"
        else:
            ext = "." + self.get("original-format")
        return ext