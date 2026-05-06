def quality(self):
        """ Can't really trust presence of a schema here, but there is an ID sometimes """
        try:
            qid = int((self.tool_metadata or {}).get("quality", 0))
        except:
            qid = 0

        # We might be able to get the quality strings from the item's tags
        internal_name, name = "normal", "Normal"
        if self.tags:
            tags = {x.get('category'): x for x in self.tags}
            if 'Quality' in tags:
                internal_name, name = tags['Quality'].get('internal_name'), tags['Quality'].get('name')

        return qid, internal_name, name