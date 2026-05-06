def export(self):
        """Export all attributes of the user to a dict.

        :return: attributes of the user.
        :rtype: dict.
        """
        data = {}
        data["name"] = self.name
        data["contributions"] = self.contributions
        data["avatar"] = self.avatar
        data["followers"] = self.followers
        data["join"] = self.join
        data["organizations"] = self.organizations
        data["repositories"] = self.numberOfRepos
        data["bio"] = self.bio
        data["private"] = self.private
        data["public"] = self.public
        data["location"] = self.location
        return data