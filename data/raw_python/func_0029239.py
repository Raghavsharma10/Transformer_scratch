def path_to_key(filepath):
        """Return the sha1sum (key) belonging to the file at filepath."""
        tmp, last = os.path.split(filepath)
        tmp, middle = os.path.split(tmp)
        return '{}{}{}'.format(os.path.basename(tmp), middle,
                               os.path.splitext(last)[0])