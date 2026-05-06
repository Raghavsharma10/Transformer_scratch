def get_browser_datetime(webdriver):
        """
        Get the current date/time on the web browser as a Python datetime object.
        This date matches 'new Date();' when ran in JavaScript console.
        Args:
            webdriver: Selenium WebDriver instance
        
        Returns: 
            datetime - Python datetime object.

        Usage::
        
            browser_datetime = WebUtils.get_browser_datetime(driver)
            local_datetime = datetime.now()
            print("Difference time difference between browser and your local machine is:",
                   local_datetime - browser_datetime)
        """
        js_stmt = """
            var wtf_get_date = new Date();
            return {'month':wtf_get_date.getMonth(), 
                    'day':wtf_get_date.getDate(), 
                    'year':wtf_get_date.getFullYear(),
                    'hours':wtf_get_date.getHours(),
                    'minutes':wtf_get_date.getMinutes(),
                    'seconds':wtf_get_date.getSeconds(),
                    'milliseconds':wtf_get_date.getMilliseconds()};
        """
        browser_date = webdriver.execute_script(js_stmt)
        return datetime(int(browser_date['year']),
                        int(browser_date['month']) + 1,  # javascript months start at 0 
                        int(browser_date['day']),
                        int(browser_date['hours']),
                        int(browser_date['minutes']),
                        int(browser_date['seconds']),
                        int(browser_date['milliseconds']))