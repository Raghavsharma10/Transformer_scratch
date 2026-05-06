def _process_info(raw_info: VideoInfo) -> VideoInfo:
        """Process raw information about the video (parse date, etc.)."""
        raw_date = raw_info.date
        date = datetime.strptime(raw_date, '%Y-%m-%d %H:%M')  # 2018-04-05 17:00
        video_info = raw_info._replace(date=date)
        return video_info