def start_disk_recording(self, filename, autoname=False):
        """Starts data recording to disk. Specify the filename without extension. 
        If autoname = True a new file will be opened for data recording each time 
        the specified time interval has elapsed. The current date and time is then 
        automatically added to the filename."""
        self.pdx.StartDiskRecording(filename, autoname)