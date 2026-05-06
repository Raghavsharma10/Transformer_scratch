def get_user_details(self, response):
        """Return user details from GitHub account"""

        account = response['account']
        metadata = json.loads(account.get('json_metadata') or '{}')
        account['json_metadata'] = metadata

        return {
            'id': account['id'],
            'username': account['name'],
            'name': metadata.get("profile", {}).get('name', ''),
            'account': account,
        }