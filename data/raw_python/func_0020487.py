def get_all_build_configs_by_labels(self, label_selectors):
        """
        Returns all builds matching a given set of label selectors. It is up to the
        calling function to filter the results.
        """
        labels = ['%s=%s' % (field, value) for field, value in label_selectors]
        labels = ','.join(labels)
        url = self._build_url("buildconfigs/", labelSelector=labels)
        return self._get(url).json()['items']