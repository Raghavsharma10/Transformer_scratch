def get_third_party(self, third_party):
        """Return the account for the given third-party.  Raise <something> if the third party doesn't belong to this bookset."""
        actual_account = third_party.get_account()
        assert actual_account.get_bookset() == self
        return ThirdPartySubAccount(actual_account, third_party=third_party)