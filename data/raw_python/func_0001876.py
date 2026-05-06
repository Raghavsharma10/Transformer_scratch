def text_to_speech(text, synthesizer, synth_args, sentence_break):
    """
    Converts given text to a pydub AudioSegment using a specified speech
    synthesizer. At the moment, IBM Watson's text-to-speech API is the only
    available synthesizer.

    :param text:
        The text that will be synthesized to audio.
    :param synthesizer:
        The text-to-speech synthesizer to use.  At the moment, 'watson' is the
        only available input.
    :param synth_args:
        A dictionary of arguments to pass to the synthesizer. Parameters for
        authorization (username/password) should be passed here.
    :param sentence_break:
        A string that identifies a sentence break or another logical break in
        the text. Necessary for text longer than 50 words. Defaults to '. '.
    """
    if len(text.split()) < 50:
        if synthesizer == 'watson':
            with open('.temp.wav', 'wb') as temp:
                temp.write(watson_request(text=text, synth_args=synth_args).content)
            response = AudioSegment.from_wav('.temp.wav')
            os.remove('.temp.wav')
            return response
        else:
            raise ValueError('"' + synthesizer + '" synthesizer not found.')
    else:
        segments = []
        for i, sentence in enumerate(text.split(sentence_break)):
            if synthesizer == 'watson':
                with open('.temp' + str(i) + '.wav', 'wb') as temp:
                    temp.write(watson_request(text=sentence, synth_args=synth_args).content)
                segments.append(AudioSegment.from_wav('.temp' + str(i) + '.wav'))
                os.remove('.temp' + str(i) + '.wav')
            else:
                raise ValueError('"' + synthesizer + '" synthesizer not found.')

        response = segments[0]
        for segment in segments[1:]:
            response = response + segment

        return response