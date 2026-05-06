def VerifyURL(cls, url):
        """ 获取当前页面的url """
          
        if Web.driver.current_url == url:
            return True
        else:
            print("VerifyURL: %s" % Web.driver.current_url)
            return False