def is_premium(self, media_type):
        """Get if the session is premium for a given media type

        @param str media_type       Should be one of ANDROID.MEDIA_TYPE_*
        @return bool
        """
        if self.logged_in:
            if media_type in self._user_data['premium']:
                return True
        return False