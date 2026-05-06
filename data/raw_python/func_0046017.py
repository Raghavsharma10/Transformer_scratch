def pull(self, conf, ignore_missing=False):
        """Push this image"""
        with Builder().remove_replaced_images(conf):
            self.push_or_pull(conf, "pull", ignore_missing=ignore_missing)