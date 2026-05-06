def shape(self):
        """
        Returns (rowCount, valueCount)
        """
        bf = self.copy()
        content = requests.get(bf.dataset_url).json()
        rowCount = content['status']['rowCount']
        valueCount = content['status']['valueCount']

        return (rowCount, valueCount)