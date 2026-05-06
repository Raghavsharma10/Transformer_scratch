def _handle_response(self, response):
        """
        internal method to throw the correct exception if something went wrong
        """
        status = response.status_code
        if status == 400:
          msg = u"bad request"
          raise exceptions.BadRequest(status, msg)
        elif status == 401:
          msg = u"authorization failed user:%s" % (self.sk_user)
          raise exceptions.Unauthorized(status, msg)
        elif status == 404:
          raise exceptions.NotFound()
        elif status == 422:
          msg = u"bad request"
          raise exceptions.BadRequest(status, msg)
        elif status in range(400, 500):
          msg = u"unexpected bad request"
          raise exceptions.BadRequest(status, msg) 
        elif status in range(500, 600):
          raise exceptions.ServerError()
        return response