def clean_videos(self):
        """
        Validates that all values in the video list are integer ids and removes all None values.
        """
        if self.videos:
            self.videos = [int(v) for v in self.videos if v is not None and is_valid_digit(v)]