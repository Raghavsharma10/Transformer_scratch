def render_to_response(self, context):
        "Return HttpResponse."
        return http.HttpResponse(
            self.render_template(context),
            content_type=self.mimetype)