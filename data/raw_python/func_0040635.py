def makeVisualSong(self):
        """Return a sequence of images and durations.
        """
        self.files=os.listdir(self.basedir)
        self.stairs=[i for i in self.files if ("stair" in i) and ("R" in i)]
        self.sectors=[i for i in self.files if "sector" in i]
        self.stairs.sort()
        self.sectors.sort()
        filenames=[self.basedir+i for i in self.sectors[:4]]
        self.iS0=mpy.ImageSequenceClip(filenames,durations=[1.5,2.5,.5,1.5])
        self.iS1=mpy.ImageSequenceClip(
                          [self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3],
                           self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3],
                           self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3],
                           self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3]],
                durations=[0.25]*8)
        self.iS2=mpy.ImageSequenceClip(
                          [self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3],
                           self.basedir+self.sectors[2],
                           self.basedir+self.sectors[3],
                           self.basedir+self.sectors[0]],
                durations=[0.75,0.25,0.75,0.25,2.]) # cai para sensível

        self.iS3=mpy.ImageSequenceClip(
                          [self.basedir+"BLANK.png",
                           self.basedir+self.sectors[0],
                           self.basedir+self.sectors[1],
                           self.basedir+self.sectors[1],
                           self.basedir+self.sectors[1],
                           self.basedir+self.sectors[0],
                           self.basedir+self.sectors[0]],
                durations=[1,0.5,2.,.25,.25,1.75, 0.25]) # [-1,8]

        self.iS4=mpy.ImageSequenceClip(
                          [self.basedir+self.sectors[2], # 1
                           self.basedir+self.sectors[3], # .5
                           self.basedir+self.sectors[5], # .5
                           self.basedir+self.sectors[2], # .75
                           self.basedir+self.sectors[0], #.25
                           self.basedir+self.sectors[2], # 1
                           self.basedir+self.sectors[0], # 2 8
                           self.basedir+self.sectors[3], # 2 7
                           self.basedir+self.sectors[0], # 2 -1
                          self.basedir+"BLANK.png",# 2
                           ],
                durations=[1,0.5,0.5,.75,
                              .25,1., 2.,2.,2.,2.]) # [0,7,11,0]

        self.iS=mpy.concatenate_videoclips((
            self.iS0,self.iS1,self.iS2,self.iS3,self.iS4))