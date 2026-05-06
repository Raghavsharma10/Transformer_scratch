def parse_path(path):
        """
        Parse a video at filepath, using pymediainfo framework.
        :param path: path of video to parse as string
        """
        import pymediainfo

        metadata = Metadata()
        log.debug('pymediainfo: parsing "{path}" ...'.format(path=path))
        parseRes = pymediainfo.MediaInfo.parse(str(path))
        log.debug('... parsing FINISHED')
        for track in parseRes.tracks:
            log.debug('... found track type: "{track_type}"'.format(track_type=track.track_type))
            if track.track_type == 'Video':
                duration_ms = track.duration
                framerate = track.frame_rate
                framecount = track.frame_count
                log.debug('mode={mode}'.format(mode=track.frame_rate_mode))
                if duration_ms is None or framerate is None:
                    log.debug('... Video track does not have duration and/or framerate.')
                    continue
                log.debug('... duration = {duration_ms} ms, framerate = {framerate} fps'.format(duration_ms=duration_ms,
                                                                                               framerate=framerate))
                metadata.add_metadata(
                    MetadataVideoTrack(
                        duration_ms=duration_ms,
                        framerate=float(framerate),
                        framecount=framecount
                    )
                )
        return metadata