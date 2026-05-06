def _getDetails(self, uriRef, associations_details):
        """
        Given a uriRef, return a dict of all the details for that Ref
        use the uriRef as the 'id' of the dict
        """
        associationDetail = {}
        for detail in associations_details:
            if detail['subject'] == uriRef:
                associationDetail[detail['predicate']] = detail['object']
            associationDetail['id'] = uriRef
        return associationDetail