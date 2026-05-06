def whisper_filename(self):
        """Build a file path to the Whisper database"""
        source_name = self.source_id and self.source.name or ''
        return get_valid_filename("{0}__{1}.wsp".format(source_name,
                                                        self.name))