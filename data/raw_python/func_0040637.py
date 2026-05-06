def makeAnimation(self):
        """Use pymovie to render (visual+audio)+text overlays.
        """
        aclip=mpy.AudioFileClip("sound.wav")
        self.iS=self.iS.set_audio(aclip)
        self.iS.write_videofile("mixedVideo.webm",15,audio=True)
        print("wrote "+"mixedVideo.webm")