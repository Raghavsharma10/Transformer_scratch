def detect_languages(self, texts):
        """
            Params:
                ::texts = Array of texts for detect languages

            Returns:
                Returns language present on array of text.
        """
        text_list = TextUtils.format_list_to_send(texts)
        infos_translate = TextDetectLanguageModel(text_list).to_dict()
        texts_for_detect = TextUtils.change_key(infos_translate, "text",
                                                    "texts", infos_translate["text"])
        mode_translate = TranslatorMode.DetectArray.value
        return self._get_content(texts_for_detect, mode_translate)