def as_dict(self):
        """
        Additionally encodes headers.

        :return:
        """
        data = super(BaseEmail, self).as_dict()
        data["Headers"] = [{"Name": name, "Value": value} for name, value in data["Headers"].items()]
        for field in ("To", "Cc", "Bcc"):
            if field in data:
                data[field] = list_to_csv(data[field])
        data["Attachments"] = [prepare_attachments(attachment) for attachment in data["Attachments"]]
        return data