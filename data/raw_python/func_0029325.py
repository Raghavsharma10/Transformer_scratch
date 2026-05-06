def set_expected_action(self, expected_response_action):
        """
        Expected outcome of the transaction ('APPROVED' or 'DECLINED')
        """
        if expected_response_action.upper() not in ['APPROVED', 'APPROVE', 'DECLINED', 'DECLINE']:
            return False

        self.expected_response_action = expected_response_action.upper()
        return True