def download_all(self, target_dir, source=None, project_data=False,
                     memberlist=None, excludelist=None,
                     max_size=MAX_SIZE_DEFAULT, id_filename=False):
        """
        Download data for all users including shared data files.

        :param target_dir: This field is the target directory to download data.
        :param source: This field is the data source. It's default value is
            None.
        :param project_data: This field is data related to particular project.
            It's default value is False.
        :param memberlist: This field is list of members whose data will be
            downloaded. It's default value is None.
        :param excludelist: This field is list of members whose data will be
            skipped. It's default value is None.
        :param max_size: This field is the maximum file size. It's default
            value is 128m.
        """
        members = self.project_data.keys()
        for member in members:
            if not (memberlist is None) and member not in memberlist:
                logging.debug('Skipping {}, not in memberlist'.format(member))
                continue
            if excludelist and member in excludelist:
                logging.debug('Skipping {}, in excludelist'.format(member))
                continue
            member_dir = os.path.join(target_dir, member)
            if not os.path.exists(member_dir):
                os.mkdir(member_dir)
            if project_data:
                self.download_member_project_data(
                    member_data=self.project_data[member],
                    target_member_dir=member_dir,
                    max_size=max_size,
                    id_filename=id_filename)
            else:
                self.download_member_shared(
                    member_data=self.project_data[member],
                    target_member_dir=member_dir,
                    source=source,
                    max_size=max_size,
                    id_filename=id_filename)