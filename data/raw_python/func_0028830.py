def welcome_page(self):
        """
        Defaulf welcome page when the route / is note mapped yet
        :rtype: HttpResponse
        """
        message = "HTTP/1.1 200 OK RINZLER FRAMEWORK"
        return HttpResponse(
            "<center><h1>{0}({1})</h1></center>".format(message, self.app.app_name),
            content_type="text/html", charset="utf-8"
        )