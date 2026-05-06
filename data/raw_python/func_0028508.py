def speak_phrase(self, text, language, format_audio=None, option=None):
        """
            This method is very similar to the above, the difference between 
            them is that this method creates an object of class 
            TranslateSpeak(having therefore different attributes) and use 
            another url, as we see the presence of SpeakMode enumerator instead
            of Translate.
            The parameter ::language:: is the same as the previous
            method(the parameter ::lang_to::). To see all possible languages go 
            to the home page of the documentation that library.
            The parameter ::format_audio:: can be of two types: "audio/mp3" or
            "audio/wav". If we do not define, Microsoft api will insert by
            default the "audio/wav". It is important to be aware that, to 
            properly name the file downloaded by AudioSpeaked
            class(which uses theclassmethod download).
            The parameter ::option:: is responsible for setting the audio quality. 
            It can be of two types: "MaxQuality" or "MinQuality". By default, if
            not define, it will be "MinQuality".
        """
        infos_speak_translate = SpeakModel(
            text, language, format_audio, option).to_dict()
        mode_translate = TranslatorMode.SpeakMode.value
        return self._get_content(infos_speak_translate, mode_translate)