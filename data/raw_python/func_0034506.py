def get_info(self) -> dict:
        """Get information about the videos from YoutubeDL package."""
        with suppress_stdout():
            with youtube_dl.YoutubeDL() as ydl:
                info_dict = ydl.extract_info(self.url, download=False)
                return info_dict