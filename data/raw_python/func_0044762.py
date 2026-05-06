def __copy_extracted(self, path, destination):
        """Copies a file that was already extracted to the destination directory.

        Args:
            path (str):
                Relative (to the root of the archive) of the file to copy.

            destination (str):
                Directory to extract the archive to.
        """

        unpacked_dir = self.filename + '.unpacked'
        if not os.path.isdir(unpacked_dir):
            LOGGER.warn(
                'Failed to copy extracted file %s, no extracted dir',
                path
            )

            return

        source_path = os.path.join(unpacked_dir, path)

        if not os.path.exists(source_path):
            LOGGER.warn(
                'Failed to copy extracted file %s, does not exist',
                path
            )

            return

        destination_path = os.path.join(destination, path)
        shutil.copyfile(source_path, destination_path)