def delete(self, name=None):
        "Delete the shelve data file."
        logger.info('clearing shelve data')
        self.close()
        for path in Path(self.create_path.parent, self.create_path.name), \
            Path(self.create_path.parent, self.create_path.name + '.db'):
            logger.debug(f'clearing {path} if exists: {path.exists()}')
            if path.exists():
                path.unlink()
                break