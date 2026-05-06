def add_policy(self, name, action, resource, subject, condition,
            policy_set_id=None, effect='PERMIT'):
        """
        Will create a new policy set to enforce the given policy details.

        The name is just a helpful descriptor for the policy.

        The action maps to a HTTP verb.

        Policies are evaluated against resources and subjects.  They are
        identified by matching a uriTemplate or attributes.

        Examples::

            resource = {
                "uriTemplate": "/asset/{id}"
                }

            subject: {
                "attributes": [{
                    "issuer": "default",
                    "name": "role"
                    }]
                }

        The condition is expected to be a string that defines a groovy
        operation that can be evaluated.

        Examples::

            condition = "match.single(subject.attributes('default', 'role'),
                'admin')

        """
        # If not given a policy set id will generate one
        if not policy_set_id:
            policy_set_id = str(uuid.uuid4())

        # Only a few operations / actions are supported in policy definitions
        if action not in ['GET', 'PUT', 'POST', 'DELETE']:
            raise ValueError("Invalid action")

        # Defines a single policy to be part of the policy set.
        policy = {
            "name": name,
            "target": {
                "resource": resource,
                "subject": subject,
                "action": action,
                },
            "conditions": [{
                "name": "",
                "condition": condition,
                }],
            "effect": effect,
        }

        # Body of the request is a list of policies
        body = {
            "name": policy_set_id,
            "policies": [policy],
        }

        result = self._put_policy_set(policy_set_id, body)
        return result