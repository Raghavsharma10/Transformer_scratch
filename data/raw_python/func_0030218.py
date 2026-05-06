def report_status(self):
        """Get status of player from mopidy core and send webhook.
        """
        current_status = {
            'current_track': self.core.playback.current_track.get(),
            'state': self.core.playback.state.get(),
            'time_position': self.core.playback.time_position.get(),
        }
        send_webhook(self.config, {'status_report': current_status})
        self.report_again(current_status)