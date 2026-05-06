def export(self, hashVal, hashPath, tags=None, galleries=None):
        """
        The export function needs to:
        - Move source image to asset folder
        - Rename to guid.ext
        - Save thumbnail, video_thumbnail, and MP4 versions.  If the source is already h264, then only transcode the thumbnails
        """

        self.source = hashPath.replace('\\', '/').replace(ROOT, '')
        galleries = galleries or []
        tags = tags or []

        # -- Get info
        videodata = self.info()
        self.width = videodata['width']
        self.height = videodata['height']
        self.framerate = videodata['framerate']
        self.duration = videodata['duration']

        self.generateThumbnail()

        for gal in galleries:
            g = Gallery.objects.get(pk=int(gal))
            g.videos.add(self)

        self.tagArtist()

        for tagName in tags:
            tag = Tag.objects.get_or_create(name=tagName)[0]
            self.tags.add(tag)

        if not self.guid:
            self.guid = self.getGuid().guid

        # -- Set the temp video while processing
        queuedvideo = VideoQueue.objects.get_or_create(video=self)[0]
        queuedvideo.save()

        self.save()

        try:
            item = VideoQueue()
            item.video = self
            item.save()
        except IntegrityError:
            # -- Already queued
            pass