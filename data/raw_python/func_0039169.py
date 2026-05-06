def from_socket(controller, host=None, port=None, track_path=None, log_level=logging.ERROR):
        """Create rocket instance using socket connector"""
        rocket = Rocket(controller, track_path=track_path, log_level=log_level)
        rocket.connector = SocketConnector(controller=controller,
                                           tracks=rocket.tracks,
                                           host=host,
                                           port=port)
        return rocket