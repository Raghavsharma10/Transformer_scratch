def load_all_from_directory(self, directory_path):
        """Return a list of dict from a directory containing files
        """
        datas = []
        for root, folders, files in os.walk(directory_path):
            for f in files:
                datas.append(self.load_from_file(os.path.join(root, f)))

        return datas