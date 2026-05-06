def parse(self, rev_string):
        """
        :param rev_string:
        :type rev_string: str
        """
        elements = rev_string.split(MESSAGE_LINE_SEPARATOR)

        heading = elements[0]

        heading_elements = heading.split(" ")

        self.revision_id = heading_elements[2]
        datetime_str = "{} {}".format(
            heading_elements[0],
            heading_elements[1]
        )
        self.release_date = datetime.datetime.strptime(
            datetime_str,
            DATETIME_FORMAT
        )

        self.description = elements[1]
        self.message = elements[2]