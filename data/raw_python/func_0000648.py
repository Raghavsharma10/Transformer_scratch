def get_page_content(self, page_id, page_info=0):
        """
          PageInfo
          0 - Returns only basic page content, without selection markup and binary data objects. This is the standard value to pass.
          1 - Returns page content with no selection markup, but with all binary data.
          2 - Returns page content with selection markup, but no binary data.
          3 - Returns page content with selection markup and all binary data.
        """
        try:
            return(self.process.GetPageContent(page_id, "", page_info))
        except Exception as e: 
            print(e)
            print("Could not get Page Content")