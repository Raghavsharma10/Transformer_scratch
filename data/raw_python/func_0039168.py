def from_project_file(controller, project_file, track_path=None, log_level=logging.ERROR):
        """Create rocket instance using project file connector"""
        rocket = Rocket(controller, track_path=track_path, log_level=log_level)
        rocket.connector = ProjectFileConnector(project_file,
                                                controller=controller,
                                                tracks=rocket.tracks)
        return rocket