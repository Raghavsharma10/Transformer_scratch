def scan(self, paths=None, depth=2):
        """scan media files in all paths
        """
        song_exts = ['mp3', 'ogg', 'wma', 'm4a']
        exts = song_exts
        paths = paths or [Library.DEFAULT_MUSIC_FOLDER]
        depth = depth if depth <= 3 else 3
        media_files = []
        for directory in paths:
            logger.debug('正在扫描目录(%s)...', directory)
            media_files.extend(scan_directory(directory, exts, depth))
        logger.info('共扫描到 %d 个音乐文件，准备将其录入本地音乐库', len(media_files))

        for fpath in media_files:
            add_song(fpath, self._songs, self._artists, self._albums)
        logger.info('录入本地音乐库完毕')