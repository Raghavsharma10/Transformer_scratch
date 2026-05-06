def speak(self, sentence):
        """ Speak a sentence using Google TTS.
        :param sentence: the sentence to speak.
        """
        temp_dir = "/tmp/"
        filename = "gtts.mp3"
        file_path = "{}/{}".format(temp_dir, filename)

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        def delete_file():
            try:
                os.remove(file_path)
                if not os.listdir(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass
            except:
                pass

        if self.logger is not None:
            self.logger.info("Google TTS: {}".format(sentence))
        tts = gTTS(text=sentence, lang=self.locale)
        tts.save(file_path)
        AudioPlayer.play_async(file_path, delete_file)