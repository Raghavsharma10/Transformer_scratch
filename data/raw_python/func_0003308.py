def start_response(self, status = 200, headers = [], clearheaders = True, disabletransferencoding = False):
        "Start to send response"
        if self._sendHeaders:
            raise HttpProtocolException('Cannot modify response, headers already sent')
        self.status = status
        self.disabledeflate = disabletransferencoding
        if clearheaders:
            self.sent_headers = headers[:]
        else:
            self.sent_headers.extend(headers)