def parent_ids(self):
        """
        Returns an array of parent Biosample IDs. If the current Biosample has a part_of relationship,
        the Biosampled referenced there will be returned. Otherwise, if the current Biosample was
        generated from a pool of Biosamples (pooled_from_biosample_ids), then those will be returned.
        Otherwise, the result will be an empty array.
        """
        action = os.path.join(self.record_url, "parent_ids")
        res = requests.get(url=action, headers=HEADERS, verify=False)
        res.raise_for_status()
        return res.json()["biosamples"]