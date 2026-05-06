def account_info(self):
        """ Certain attributes have a user's account information
        associated with it such as a gifted or crafted item.

        A dict with two keys: 'persona' and 'id64'.
        None if the attribute has no account information attached to it. """
        account_info = self._attribute.get("account_info")
        if account_info:
            return {"persona": account_info.get("personaname", ""),
                    "id64": account_info["steamid"]}
        else:
            return None