def get_failure_reason(self):
        """
        Find the reason a pod failed

        :return: dict, which will always have key 'reason':
                 reason: brief reason for state
                 containerID (if known): ID of container
                 exitCode (if known): numeric exit code
        """

        reason_key = 'reason'
        cid_key = 'containerID'
        exit_key = 'exitCode'

        pod_status = self.json.get('status', {})
        statuses = pod_status.get('containerStatuses', [])

        # Find the first non-zero exit code from a container
        # and return its 'message' or 'reason' value
        for status in statuses:
            try:
                terminated = status['state']['terminated']
                exit_code = terminated['exitCode']
                if exit_code != 0:
                    reason_dict = {
                        exit_key: exit_code,
                    }

                    if 'containerID' in terminated:
                        reason_dict[cid_key] = terminated['containerID']

                    for key in ['message', 'reason']:
                        try:
                            reason_dict[reason_key] = terminated[key]
                            break
                        except KeyError:
                            continue
                    else:
                        # Both 'message' and 'reason' are missing
                        reason_dict[reason_key] = 'Exit code {code}'.format(
                            code=exit_code
                        )

                    return reason_dict
            except KeyError:
                continue

        # Failing that, return the 'message' or 'reason' value for the
        # pod
        for key in ['message', 'reason']:
            try:
                return {reason_key: pod_status[key]}
            except KeyError:
                continue

        return {reason_key: pod_status['phase']}