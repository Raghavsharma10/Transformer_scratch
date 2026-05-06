def create_new_page (self, section_id, new_page_style=0):
        """
          NewPageStyle
          0 - Create a Page that has Default Page Style
          1 - Create a blank page with no title
          2 - Createa blank page that has no title
        """
        try:
            self.process.CreateNewPage(section_id, "", new_page_style)
        except Exception as e: 
            print(e)
            print("Unable to create the page")