def log(self, message: str):
        """
        @deprecated

        :param message:
        :return:

        """
        dset_log_id = '_%s_log' % self.iid

        if dset_log_id not in self.parent.data.keys():
            dset = self.parent.data.create_dataset(
                dset_log_id, shape=(1,),
                dtype=np.dtype([
                    ('dt_log', '<i8'),
                    ('message', 'S250')
                ])
            )
        else:
            dset = self.parent.data[dset_log_id]

        timestamp = np.array(
            datetime.now().strftime("%s")
        ).astype('<i8').view('<M8[s]')

        dset['dt_log'] = timestamp.view('<i8')
        dset['message'] = message
        self.parent.data.flush()