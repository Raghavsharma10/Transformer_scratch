def makeImages(self):
        """Make spiral images in sectors and steps.

        Plain, reversed,
        sectorialized, negative sectorialized
        outline, outline reversed, lonely
        only nodes, only edges, both
        """
        # make layout
        self.makeLayout()
        self.setAgraph()
        # make function that accepts a mode, a sector
        # and nodes and edges True and False
        self.plotGraph()
        self.plotGraph("reversed",filename="tgraphR.png")
        agents=n.concatenate(self.np.sectorialized_agents__)
        for i, sector in enumerate(self.np.sectorialized_agents__):
            self.plotGraph("plain",   sector,"sector{:02}.png".format(i))
            self.plotGraph("reversed",sector,"sector{:02}R.png".format(i))
            self.plotGraph("plain", n.setdiff1d(agents,sector),"sector{:02}N.png".format(i))
            self.plotGraph("reversed",n.setdiff1d(agents,sector),"sector{:02}RN.png".format(i))
        self.plotGraph("plain",   [],"BLANK.png")