def _is_video(filepath) -> bool:
    """Check filename extension to see if it's a video file."""
    if os.path.exists(filepath):  # Could be broken symlink
        extension = os.path.splitext(filepath)[1]
        return extension in ('.mkv', '.mp4', '.avi')
    else:
        return False