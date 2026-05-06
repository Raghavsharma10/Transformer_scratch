def put_templates(self, ignore=None):
        """Yield tuple with registered template and response from client."""
        ignore = ignore or []

        def _replace_prefix(template_path, body):
            """Replace index prefix in template request body."""
            pattern = '__SEARCH_INDEX_PREFIX__'

            prefix = self.app.config['SEARCH_INDEX_PREFIX'] or ''
            if prefix:
                assert pattern in body, "You are using the prefix `{0}`, "
                "but the template `{1}` does not contain the "
                "pattern `{2}`.".format(prefix, template_path, pattern)

            return body.replace(pattern, prefix)

        def _put_template(template):
            """Put template in search client."""
            with open(self.templates[template], 'r') as fp:
                body = fp.read()
                replaced_body = _replace_prefix(self.templates[template], body)
                return self.templates[template],\
                    current_search_client.indices.put_template(
                        name=template,
                        body=json.loads(replaced_body),
                        ignore=ignore,
                )

        for template in self.templates:
            yield _put_template(template)