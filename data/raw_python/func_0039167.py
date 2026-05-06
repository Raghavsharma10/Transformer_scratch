def from_files(controller, track_path, log_level=logging.ERROR):
        """Create rocket instance using project file connector"""
        rocket = Rocket(controller, track_path=track_path, log_level=log_level)
        rocket.connector = FilesConnector(track_path,
                                          controller=controller,
                                          tracks=rocket.tracks)
        return rocket