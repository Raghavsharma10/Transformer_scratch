def files_ondisk(self, file_objs: models.File) -> set:
        """Returns a list of files that are not on disk."""

        return set([ file_obj for file_obj in file_objs if Path(file_obj.full_path).is_file() ])