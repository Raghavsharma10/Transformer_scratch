def names(cls):
        """A list of all emoji names without file extension."""
        if not cls._files:
            for f in os.listdir(cls._image_path):
                if(not f.startswith('.') and
                   os.path.isfile(os.path.join(cls._image_path, f))):
                    cls._files.append(os.path.splitext(f)[0])

        return cls._files