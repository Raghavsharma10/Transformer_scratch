def evt_toggle_pause(self, *args):  # pylint: disable=unused-argument
        """Pauses and resumes the video source."""
        if self.event_source._timer is None:  # noqa: e501 pylint: disable=protected-access
            self.event_source.start()
        else:
            self.event_source.stop()