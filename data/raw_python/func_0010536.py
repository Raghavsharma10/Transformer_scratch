def matches_video_filename(self, video):
        """
        Detect whether the filename of videofile matches with this SubtitleFile.
        :param video: VideoFile instance
        :return: True if match
        """

        vid_fn = video.get_filename()
        vid_base, _ = os.path.splitext(vid_fn)
        vid_base = vid_base.lower()

        sub_fn = self.get_filename()
        sub_base, _ = os.path.splitext(sub_fn)
        sub_base = sub_base.lower()

        log.debug('matches_filename(subtitle="{sub_filename}", video="{vid_filename}") ...'.format(
            sub_filename=sub_fn, vid_filename=vid_fn))

        matches = sub_base == vid_base

        lang = None
        if not matches:
            if sub_base.startswith(vid_base):
                sub_rest = sub_base[len(vid_base):]
                while len(sub_rest) > 0:
                    if sub_rest[0].isalnum():
                        break
                    sub_rest = sub_rest[1:]
                try:
                    lang = Language.from_unknown(sub_rest, xx=True, xxx=True)
                    matches = True
                except NotALanguageException:
                    matches = False

        if matches:
            log.debug('... matches (language={language})'.format(language=lang))
        else:
            log.debug('... does not match')
        return matches