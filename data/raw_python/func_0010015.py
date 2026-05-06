def row_to_dictionary(header_row_web_element, row_webelement):
        """
        Converts a row into a dictionary of key/values.
        (Note: assumes all rows/columns have uniform cells.  Does not 
        account for any row or column spans)
        
        Args:
            header_row_web_element (WebElement): WebElement reference to the column headers.
            row_webelement (WebElement): WebElement reference to row.
        
        Returns:
            Returns a dictionary object containing keys consistenting of the column headers 
            and values consisting of the row contents.

        Usage::

            self.webdriver.get("http://the-internet.herokuapp.com/tables")

            header = self.webdriver.find_element_by_css_selector("#table1 thead tr")
            target_row = self.webdriver.find_element_by_css_selector("#table1 tbody tr")
    
            row_values = WebUtils.row_to_dictionary(header, target_row)
            row_values ==    {'Last Name': 'Smith', 
                              'Due': '$50.00',
                              'First Name': 'John', 
                              'Web Site': 'http://www.jsmith.com', 
                              'Action': 'edit delete', 
                              'Email': 'jsmith@gmail.com'}

        """
        headers = header_row_web_element.find_elements_by_tag_name("th")
        data_cells = row_webelement.find_elements_by_tag_name("td")
        
        value_dictionary = {}
        for i in range(len(data_cells)):
            value_dictionary[ headers[i].text ] = data_cells[i].text

        return value_dictionary