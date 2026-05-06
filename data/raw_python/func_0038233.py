def save(self, *args, **kwargs):
        """sets uuid for url

        :param args: inline arguments (optional)
        :param kwargs: keyword arguments (optional)
        :return: `super.save()`
        """
        if not self.id:  # this is a totally new instance, create uuid value
            self.url_uuid = str(uuid.uuid4()).replace("-", "")
        super(ObfuscatedUrlInfo, self).save(*args, **kwargs)