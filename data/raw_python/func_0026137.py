def _setSampleSizeBytes(self):
        """
        updates the current record of the packet size per sample and the relationship between this and the fifo reads. 
        """
        self.sampleSizeBytes = self.getPacketSize()
        if self.sampleSizeBytes > 0:
            self.maxBytesPerFifoRead = (32 // self.sampleSizeBytes)