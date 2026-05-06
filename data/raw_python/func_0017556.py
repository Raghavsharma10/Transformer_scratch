def ids(self, content_ids, object_ids):
        """Appends the content and object ids how RETS expects them"""
        result = []

        content_ids = self.split(content_ids, False)
        object_ids = self.split(object_ids)

        for cid in content_ids:
            result.append('{}:{}'.format(cid, ':'.join(object_ids)))

        return result