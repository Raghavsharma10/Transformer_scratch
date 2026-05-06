def get_edit_url(
        self,
        md_id: str = None,
        md_type: str = None,
        owner_id: str = None,
        tab: str = "identification",
    ):
        """Constructs the edition URL of a metadata.

        :param str md_id: metadata/resource UUID
        :param str owner_id: owner UUID
        :param str tab: target tab in the web form
        """
        # checks inputs
        if not checker.check_is_uuid(md_id) or not checker.check_is_uuid(owner_id):
            raise ValueError("One of md_id or owner_id is not a correct UUID.")
        else:
            pass
        if checker.check_edit_tab(tab, md_type=md_type):
            pass
        # construct URL
        return (
            "{}"
            "/groups/{}"
            "/resources/{}"
            "/{}".format(self.APP_URLS.get(self.platform), owner_id, md_id, tab)
        )