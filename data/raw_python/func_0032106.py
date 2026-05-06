def parse_template(self, template, **context):
        """
        To parse a template and return all the blocks
        """
        required_blocks = ["subject", "body"]
        optional_blocks = ["text_body", "html_body", "return_path", "format"]

        if self.template_context:
            context = dict(self.template_context.items() + context.items())
        blocks = self.template.render_blocks(template, **context)

        for rb in required_blocks:
            if rb not in blocks:
                raise AttributeError("Template error: block '%s' is missing from '%s'" % (rb, template))

        mail_params = {
            "subject": blocks["subject"].strip(),
            "body": blocks["body"]
        }
        for ob in optional_blocks:
            if ob in blocks:
                if ob == "format" and mail_params[ob].lower() not in ["html", "text"]:
                    continue
                mail_params[ob] = blocks[ob]
        return mail_params