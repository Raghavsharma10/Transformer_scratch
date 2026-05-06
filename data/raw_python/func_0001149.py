def _create_attachment(self, filename, content, mimetype=None):
        """Convert the filename, content, mimetype triple to attachment."""
        if mimetype is None:
            mimetype, _ = mimetypes.guess_type(filename)
            if mimetype is None:
                mimetype = DEFAULT_ATTACHMENT_MIME_TYPE
        basetype, subtype = mimetype.split("/", 1)
        if basetype == "text":
            attachment = SafeMIMEText(
                smart_bytes(content, DEFAULT_CHARSET), subtype, DEFAULT_CHARSET
            )
        else:
            # Encode non-text attachments with base64.
            attachment = MIMEBase(basetype, subtype)
            attachment.set_payload(content)
            encode_base64(attachment)
        if filename:
            attachment.add_header(
                "Content-Disposition", "attachment", filename=filename
            )
        return attachment