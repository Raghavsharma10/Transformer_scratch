def move(self, destination_path):
    """
    <Purpose>
      Copies 'self.temporary_file' to a non-temp file at 'destination_path' and
      closes 'self.temporary_file' so that it is removed.

    <Arguments>
      destination_path:
        Path to store the file in.

    <Exceptions>
      None.

    <Return>
      None.
    """

    self.flush()
    self.seek(0)
    destination_file = open(destination_path, 'wb')
    shutil.copyfileobj(self.temporary_file, destination_file)
    # Force the destination file to be written to disk from Python's internal
    # and the operation system's buffers.  os.fsync() should follow flush().
    destination_file.flush()
    os.fsync(destination_file.fileno())
    destination_file.close()

    # 'self.close()' closes temporary file which destroys itself.
    self.close_temp_file()