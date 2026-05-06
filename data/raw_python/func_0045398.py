def get_usage_from_mdoc(self):
        """
        get_usage_from_mdoc
        """
        usage = self.m_doc.strip().split("Usage:")

        if len(usage) > 1:
            usage = "\033[34mUsage:\033[34m" + usage[1]

        return "\n".join(usage.strip().split("\n")[:2]) + "\033[0m"