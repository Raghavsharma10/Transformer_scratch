def update_data(self):
        """
        Returns data for all users including shared data files.
        """
        url = ('https://www.openhumans.org/api/direct-sharing/project/'
               'members/?access_token={}'.format(self.master_access_token))
        results = get_all_results(url)
        self.project_data = dict()
        for result in results:
            self.project_data[result['project_member_id']] = result
            if len(result['data']) < result['file_count']:
                member_data = get_page(result['exchange_member'])
                final_data = member_data['data']
                while member_data['next']:
                    member_data = get_page(member_data['next'])
                    final_data = final_data + member_data['data']
                self.project_data[
                    result['project_member_id']]['data'] = final_data
        return self.project_data