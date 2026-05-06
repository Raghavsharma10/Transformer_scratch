def translate_links(self, text, in_comment=None):
        """
        Turn all @link tags in `text` into HTML anchor tags.

        `in_comment` is the `CommentDoc` that contains the text, for
        relative method lookups.
        """
        def replace_link(matchobj):
            ref = matchobj.group(1)
            return '<a href = "%s">%s</a>' % (
                    self.translate_ref_to_url(ref, in_comment), ref)
        return re.sub('{@link ([\w#]+)}', replace_link, text)