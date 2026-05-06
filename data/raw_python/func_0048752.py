def emit(self,rlen=150):
      """Emit a read based on a source sequence"""
      source_tx = self._source.emit()
      source_read = self._cutter.cut(source_tx)
      if self._flip and self.options.rand.random() < 0.5: source_read = source_read.rc()
      srname = self.options.rand.uuid4()
      seqfull = FASTQ('@'+self.options.rand.uuid4()+"\tlong\n"+str(source_read.sequence)+"\n+\n"+'I'*source_read.sequence.length+"\n")
      seqperm1 = seqfull.copy()
      seqperm2 = seqfull.copy()
      for e in self.errors:
        seqperm1 = e.permute(seqperm1)
        seqperm2 = e.permute(seqperm2)
      sleft = seqperm1[0:rlen]
      sleft = FASTQ('@'+sleft.name+"\tleft\n"+sleft.sequence+"\n+\n"+sleft.qual+"\n")
      sright = seqperm2.rc()[0:rlen]
      sright = FASTQ('@'+sright.name+"\tright\n"+sright.sequence+"\n+\n"+sright.qual+"\n")
      emission = TranscriptEmission(source_tx,
             Source(source_read,
                    source_read.slice_sequence(0,rlen),
                    source_read.rc().slice_sequence(0,rlen)),
             Read(seqperm1,
                  sleft,
                  sright
                 ))
      return emission