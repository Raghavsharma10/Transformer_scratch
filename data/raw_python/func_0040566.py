def to_markdown(self):
        """
        :return:
        :rtype: str
        """
        return "## {} {}\n\n{}\n\n{}\n\n".format(
            self.release_date.strftime(DATETIME_FORMAT),
            self.revision_id,
            self.description,
            self.message
        )