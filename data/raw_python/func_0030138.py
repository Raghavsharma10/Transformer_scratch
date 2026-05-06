def is_local(self):
        """Return true is the partition file is local"""
        from ambry.orm.exc import NotFoundError
        try:
            if self.local_datafile.exists:
                return True
        except NotFoundError:
            pass

        return False