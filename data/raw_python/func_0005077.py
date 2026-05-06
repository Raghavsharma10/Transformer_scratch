def share_extender(self, share: dict, results_filtered: dict):
        """Extend share model with additional informations.

        :param dict share: share returned by API
        :param dict results_filtered: filtered search result
        """
        # add share administration URL
        creator_id = share.get("_creator").get("_tag")[6:]
        share["admin_url"] = "{}/groups/{}/admin/shares/{}".format(
            self.app_url, creator_id, share.get("_id")
        )
        # check if OpenCatalog is activated
        opencat_url = "{}/s/{}/{}".format(
            self.oc_url, share.get("_id"), share.get("urlToken")
        )
        if requests.head(opencat_url):
            share["oc_url"] = opencat_url
        else:
            pass
        # add metadata ids list
        share["mds_ids"] = (i.get("_id") for i in results_filtered)

        return share