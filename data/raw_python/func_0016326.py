def prepare_attachments(attachment):
    """
    Converts incoming attachment into dictionary.
    """
    if isinstance(attachment, tuple):
        result = {"Name": attachment[0], "Content": attachment[1], "ContentType": attachment[2]}
        if len(attachment) == 4:
            result["ContentID"] = attachment[3]
    elif isinstance(attachment, MIMEBase):
        payload = attachment.get_payload()
        content_type = attachment.get_content_type()
        # Special case for message/rfc822
        # Even if RFC implies such attachments being not base64-encoded,
        # Postmark requires all attachments to be encoded in this way
        if content_type == "message/rfc822" and not isinstance(payload, str):
            payload = b64encode(payload[0].get_payload(decode=True)).decode()
        result = {
            "Name": attachment.get_filename() or "attachment.txt",
            "Content": payload,
            "ContentType": content_type,
        }
        content_id = attachment.get("Content-ID")
        if content_id:
            if content_id.startswith("<") and content_id.endswith(">"):
                content_id = content_id[1:-1]
            if (attachment.get("Content-Disposition") or "").startswith("inline"):
                content_id = "cid:%s" % content_id
            result["ContentID"] = content_id
    elif isinstance(attachment, str):
        content_type = guess_content_type(attachment)
        filename = os.path.basename(attachment)
        with open(attachment, "rb") as fd:
            data = fd.read()
        result = {"Name": filename, "Content": b64encode(data).decode("utf-8"), "ContentType": content_type}
    else:
        result = attachment
    return result