def _get_member_file_data(member_data, id_filename=False):
        """
        Helper function to get file data of member of a project.

        :param member_data: This field is data related to member in a project.
        """
        file_data = {}
        for datafile in member_data['data']:
            if id_filename:
                basename = '{}.{}'.format(datafile['id'], datafile['basename'])
            else:
                basename = datafile['basename']
            if (basename not in file_data or
                    arrow.get(datafile['created']) >
                    arrow.get(file_data[basename]['created'])):
                file_data[basename] = datafile
        return file_data