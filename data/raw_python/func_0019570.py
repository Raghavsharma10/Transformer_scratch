def to_img(self, fname, fmt="PNG", add_left=0, seqlogo=None, height=6):
        """Create a sequence logo using seqlogo.

        Create a sequence logo and save it to a file. Valid formats are: PNG, 
        EPS, GIF and PDF. 

        Parameters
        ----------
        fname : str
            Output filename.
        fmt : str , optional
            Output format (case-insensitive). Valid formats are PNG, EPS, GIF 
            and PDF.
        add_left : int , optional
            Pad motif with empty positions on the left side.
        seqlogo : str
            Location of the seqlogo executable. By default the seqlogo version 
            that is included with GimmeMotifs is used.
        height : float
            Height of the image
        """
        if not seqlogo:
            seqlogo = self.seqlogo
        if not seqlogo:
            raise ValueError("seqlogo not specified or configured")
        
        #TODO: split to_align function
        
        VALID_FORMATS = ["EPS", "GIF", "PDF", "PNG"]
        N = 1000
        fmt = fmt.upper()
        if not fmt in VALID_FORMATS:
            sys.stderr.write("Invalid motif format\n")
            return
        
        if fname[-4:].upper() == (".%s" % fmt):
            fname = fname[:-4]
        seqs = []
        if add_left == 0:
            seqs = ["" for i in range(N)]
        else:
            for nuc in ["A", "C", "T", "G"]:
                seqs += [nuc * add_left for i in range(N // 4)]

        for pos in range(len(self.pwm)):
            vals = [self.pwm[pos][0] * N]
            for i in range(1,4):
                vals.append(vals[i-1] + self.pwm[pos][i] * N)
            if vals[3] - N != 0:
                #print "Motif weights don't add up to 1! Error of %s%%" % ((vals[3] - n)/ n * 100)
                vals[3] = N
            for i in range(N):
                if i <= vals[0]:
                    seqs[i] += "A"
                elif i <= vals[1]:
                    seqs[i] += "C"
                elif i <= vals[2]:
                    seqs[i] += "G"
                elif i <= vals[3]:
                    seqs[i] += "T"
    
        f = NamedTemporaryFile(mode="w", dir=mytmpdir())
        for seq in seqs:
            f.write("%s\n" % seq)
        f.flush()
        makelogo = "{0} -f {1} -F {2} -c -a -h {3} -w {4} -o {5} -b -n -Y" 
        cmd = makelogo.format(
                              seqlogo, 
                              f.name, 
                              fmt, 
                              height,
                              len(self) + add_left, 
                              fname)
        sp.call(cmd, shell=True)