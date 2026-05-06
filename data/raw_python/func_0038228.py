def is_published(self):
        """determines if the content is/should be live

        :return: `bool`
        """
        if self.published:
            now = timezone.now()
            if now >= self.published:
                return True
        return False