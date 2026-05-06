def quality_comparator(video_data):
        """Custom comparator used to choose the right format based on the resolution."""
        def parse_resolution(res: str) -> Tuple[int, ...]:
            return tuple(map(int, res.split('x')))

        raw_resolution = video_data['resolution']
        resolution = parse_resolution(raw_resolution)
        return resolution