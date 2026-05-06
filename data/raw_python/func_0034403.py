def export(self, hashVal=None, hashPath=None, tags=None, galleries=None):
        """
        The export function needs to:
        - Move source image to asset folder
        - Rename to guid.ext
        - Save thumbnail, small, and image versions
        """
        hashVal = hashVal or self.hash
        hashPath = hashPath or self.parent / hashVal + self.ext
        
        source = hashPath.replace('\\', '/').replace(ROOT, '')
        galleries = galleries or []
        tags = tags or []

        imagefile = ROOT / source

        if imagefile.ext == '.psd':
            psd = psd_tools.PSDImage.load(imagefile)
            workImage = psd.as_PIL()
        else:
            workImage = pilImage.open(imagefile)
            self.source = source

        if imagefile.ext in ('.tif', '.tiff', '.psd'):
            png = imagefile.parent / '{}.png'.format(imagefile.namebase)
            workImage.save(png)
            workImage = pilImage.open(png)
            imagefile.move(imagefile.replace(self.hash, self.title))
            self.source = png.replace(ROOT, '')

        formats = [
            ('image', FROG_IMAGE_SIZE_CAP),
            ('small', FROG_IMAGE_SMALL_SIZE),
        ]
        for i, n in enumerate(formats):
            if workImage.size[0] > n[1] or workImage.size[1] > n[1]:
                workImage.thumbnail((n[1], n[1]), pilImage.ANTIALIAS)
                setattr(self, n[0], self.source.name.replace(hashVal, '_' * i + hashVal))
                workImage.save(ROOT + getattr(self, n[0]).name)
            else:
                setattr(self, n[0], self.source)

        self.generateThumbnail()

        for gal in galleries:
            g = Gallery.objects.get(pk=int(gal))
            g.images.add(self)

        self.tagArtist()

        for tagName in tags:
            tag = Tag.objects.get_or_create(name=tagName)[0]
            self.tags.add(tag)

        if not self.guid:
            self.guid = self.getGuid().guid
        self.save()