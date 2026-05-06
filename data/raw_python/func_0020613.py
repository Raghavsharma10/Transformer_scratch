def html_doc(self):
        """
        :returns: the lxml processed html document
        :rtype: ``lxml.html.document_fromstring`` output
        """
        
        if self.__lx_doc is None:
            cn = NHLCn()
          
            if hasattr(cn, self.report_type):
                html = getattr(cn, self.report_type)(self.game_key)
            else:
                raise ValueError('Invalid report type: %s' % self.report_type)
          
            if cn.req_err is None:
                self.__lx_doc = fromstring(html)
            else:
                self.req_err = cn.req_err
            
        return self.__lx_doc