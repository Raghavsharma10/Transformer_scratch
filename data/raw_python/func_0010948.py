def full_name(self):
        """
        The full  name of the item, generated depending
        on things such as its quality, rank, the schema language,
        and so on.
        """
        qid, quality_str, pretty_quality_str = self.quality
        custom_name = self.custom_name
        item_name = self.name
        english = (self._language == "en_US")
        rank = self.rank
        prefixed = self._schema_item.get("proper_name", False)
        prefix = ''
        suffix = ''
        pfinal = ''

        if item_name.startswith("The ") and prefixed:
            item_name = item_name[4:]

        if quality_str != "unique" and quality_str != "normal":
            pfinal = pretty_quality_str

        if english:
            if prefixed:
                if quality_str == "unique":
                    pfinal = "The"
            elif quality_str == "unique":
                pfinal = ''

        if rank and quality_str == "strange":
            pfinal = rank["name"]

        if english:
            prefix = pfinal
        elif pfinal:
            suffix = '(' + pfinal + ') ' + suffix

        return (prefix + " " + item_name + " " + suffix).strip()