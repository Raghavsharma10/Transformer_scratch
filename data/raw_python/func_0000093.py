def _embed(self, request, response):
        """Embed Chosen.js directly in html of the response."""
        if self._match(request, response):
            # Render the <link> and the <script> tags to include Chosen.
            head = render_to_string(
                "chosenadmin/_head_css.html",
                {"chosen_css": self._chosen_css()}
            )
            body = render_to_string(
                "chosenadmin/_script.html",
                {"chosen_js": self._chosen_js()}
            )

            # Re-write the Response's content to include our new html
            content = response.rendered_content
            content = content.replace('</head>', head)
            content = content.replace('</body>', body)
            response.content = content
        return response