def _create(self):
        """Create the Whisper file on disk"""
        if not os.path.exists(settings.SALMON_WHISPER_DB_PATH):
            os.makedirs(settings.SALMON_WHISPER_DB_PATH)
        archives = [whisper.parseRetentionDef(retentionDef)
                    for retentionDef in settings.ARCHIVES.split(",")]
        whisper.create(self.path, archives,
                       xFilesFactor=settings.XFILEFACTOR,
                       aggregationMethod=settings.AGGREGATION_METHOD)