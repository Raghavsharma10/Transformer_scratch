def generateThumbnail(self):
        """Generates a square thumbnail"""
        source = ROOT / self.source.name
        thumbnail = source.parent / '_{}.jpg'.format(source.namebase)

        # -- Save thumbnail and put into queue
        poster = source.parent / '__{}.jpg'.format(source.namebase)
        cmd = [FROG_FFMPEG, '-i', str(source), '-ss', '1', '-vframes', '1', str(thumbnail), '-y']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc.communicate()
        image = pilImage.open(thumbnail)
        image.save(poster)
        self.poster = poster.replace(ROOT, '')

        box, width, height = cropBox(self.width, self.height)

        # Resize
        image.thumbnail((width, height), pilImage.ANTIALIAS)
        # Crop from center
        box = cropBox(*image.size)[0]
        image = image.crop(box)
        # save
        self.thumbnail = thumbnail.replace(ROOT, '')
        image.save(thumbnail)