def invite_other_parties(self, possible_owners):
        """
        Invites the next lane's (possible) owner(s) to participate
        """
        signals.lane_user_change.send(sender=self.user,
                                      current=self,
                                      old_lane=self.old_lane,
                                      possible_owners=possible_owners
                                      )