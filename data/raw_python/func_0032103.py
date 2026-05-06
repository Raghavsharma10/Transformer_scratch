def init_app(self, app):
        """
        For Flask using the app config
        """
        self.__init__(aws_access_key_id=app.config.get("SES_AWS_ACCESS_KEY"),
                      aws_secret_access_key=app.config.get("SES_AWS_SECRET_KEY"),
                      region=app.config.get("SES_REGION", "us-east-1"),
                      sender=app.config.get("SES_SENDER", None),
                      reply_to=app.config.get("SES_REPLY_TO", None),
                      template=app.config.get("SES_TEMPLATE", None),
                      template_context=app.config.get("SES_TEMPLATE_CONTEXT", {})
        )