def get_media(self, v14_scriptable):
        """Return (images, sounds)"""
        images = []
        sounds = []
        for media in v14_scriptable.media:
            if media.class_name == 'SoundMedia':
                sounds.append(media)
            elif media.class_name == 'ImageMedia':
                images.append(media)
        return (images, sounds)