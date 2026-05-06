def bsfile(self, path):
        """Return a Build Source file ref, creating a new one if the one requested does not exist"""
        from sqlalchemy.orm.exc import NoResultFound
        from ambry.orm.exc import NotFoundError

        try:

            f =  object_session(self)\
                .query(File)\
                .filter(File.d_vid == self.vid)\
                .filter(File.major_type == File.MAJOR_TYPE.BUILDSOURCE)\
                .filter(File.path == path)\
                .one()

            return f

        except NoResultFound:
            raise NotFoundError("Failed to find file for path '{}' ".format(path))