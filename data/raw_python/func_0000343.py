def revisionId(self):
        """
        revisionId differs from id, it is details of implementation use self.id
        :return: RevisionId
        """
        log.warning("'RevisionId' requested, ensure that you are don't need 'id'")
        revision_id = self.json()['revisionId']
        assert revision_id == self.id, "RevisionId differs id-{}!=revisionId-{}".format(self.id, revision_id)
        return revision_id