def is_allowed(self, subject_id, action, resource_id, policy_sets=[]):
        """
        Evaluate a policy-set against a subject and resource.

        example/

            is_allowed('/user/j12y', 'GET', '/asset/12')

        """
        body = {
            "action": action,
            "subjectIdentifier": subject_id,
            "resourceIdentifier": resource_id,
        }

        if policy_sets:
            body['policySetsEvaluationOrder'] = policy_sets

        # Will return a 200 with decision
        uri = self.uri + '/v1/policy-evaluation'

        logging.debug("URI=" + str(uri))
        logging.debug("BODY=" + str(body))

        response = self.service._post(uri, body)

        if 'effect' in response:
            if response['effect'] in ['NOT_APPLICABLE', 'PERMIT']:
                return True

        return False