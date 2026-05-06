def delete(self, name):
        """
        Deletes the named entry.
        :param name: the entry.
        :return: the deleted entry.
        """
        i, entry = next(((i, x) for i, x in enumerate(self._uploadCache) if x['name'] == name), (None, None))
        if entry is not None:
            logger.info("Deleting " + name)
            os.remove(str(entry['path']))
            del self._uploadCache[i]
            return entry
        else:
            logger.info("Unable to delete " + name + ", not found")
            return None