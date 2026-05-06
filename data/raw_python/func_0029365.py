def close(self):
        """
        Finalises the compressed version of the spreadsheet. If you aren't using the context manager ('with' statement,
        you must call this manually, it is not triggered automatically like on a file object.
        :return: Nothing.
        """
        self.zipf.writestr("content.xml", self.dom.toxml().encode("utf-8"))
        self.zipf.close()