def metadata(self):
        """Access process configuarion values as attributes. """
        from ambry.metadata.schema import Top  # cross-module import
        top = Top()
        top.build_from_db(self.dataset)
        return top