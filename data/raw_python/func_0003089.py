def text(self,txt):
        """ Print Utf8 encoded alpha-numeric text """
        if not txt:
            return
        try:
            txt = txt.decode('utf-8')
        except:
            try:
                txt = txt.decode('utf-16')
            except:
                pass

        self.extra_chars = 0
        
        def encode_char(char):  
            """ 
            Encodes a single utf-8 character into a sequence of 
            esc-pos code page change instructions and character declarations 
            """ 
            char_utf8 = char.encode('utf-8')
            encoded  = ''
            encoding = self.encoding # we reuse the last encoding to prevent code page switches at every character
            encodings = {
                    # TODO use ordering to prevent useless switches
                    # TODO Support other encodings not natively supported by python ( Thai, Khazakh, Kanjis )
                    'cp437': TXT_ENC_PC437,
                    'cp850': TXT_ENC_PC850,
                    'cp852': TXT_ENC_PC852,
                    'cp857': TXT_ENC_PC857,
                    'cp858': TXT_ENC_PC858,
                    'cp860': TXT_ENC_PC860,
                    'cp863': TXT_ENC_PC863,
                    'cp865': TXT_ENC_PC865,
                    'cp866': TXT_ENC_PC866,
                    'cp862': TXT_ENC_PC862,
                    'cp720': TXT_ENC_PC720,
                    'cp936': TXT_ENC_PC936,
                    'iso8859_2': TXT_ENC_8859_2,
                    'iso8859_7': TXT_ENC_8859_7,
                    'iso8859_9': TXT_ENC_8859_9,
                    'cp1254'   : TXT_ENC_WPC1254,
                    'cp1255'   : TXT_ENC_WPC1255,
                    'cp1256'   : TXT_ENC_WPC1256,
                    'cp1257'   : TXT_ENC_WPC1257,
                    'cp1258'   : TXT_ENC_WPC1258,
                    'katakana' : TXT_ENC_KATAKANA,
            }
            remaining = copy.copy(encodings)

            if not encoding :
                encoding = 'cp437'

            while True: # Trying all encoding until one succeeds
                try:
                    if encoding == 'katakana': # Japanese characters
                        if jcconv:
                            # try to convert japanese text to a half-katakanas 
                            kata = jcconv.kata2half(jcconv.hira2kata(char_utf8))
                            if kata != char_utf8:
                                self.extra_chars += len(kata.decode('utf-8')) - 1
                                # the conversion may result in multiple characters
                                return encode_str(kata.decode('utf-8')) 
                        else:
                             kata = char_utf8
                        
                        if kata in TXT_ENC_KATAKANA_MAP:
                            encoded = TXT_ENC_KATAKANA_MAP[kata]
                            break
                        else: 
                            raise ValueError()
                    else:
                        encoded = char.encode(encoding)
                        break

                except ValueError: #the encoding failed, select another one and retry
                    if encoding in remaining:
                        del remaining[encoding]
                    if len(remaining) >= 1:
                        encoding = remaining.items()[0][0]
                    else:
                        encoding = 'cp437'
                        encoded  = '\xb1'    # could not encode, output error character
                        break;

            if encoding != self.encoding:
                # if the encoding changed, remember it and prefix the character with
                # the esc-pos encoding change sequence
                self.encoding = encoding
                encoded = encodings[encoding] + encoded

            return encoded
        
        def encode_str(txt):
            buffer = ''
            for c in txt:
                buffer += encode_char(c)
            return buffer

        txt = encode_str(txt)

        # if the utf-8 -> codepage conversion inserted extra characters, 
        # remove double spaces to try to restore the original string length
        # and prevent printing alignment issues
        while self.extra_chars > 0: 
            dspace = txt.find('  ')
            if dspace > 0:
                txt = txt[:dspace] + txt[dspace+1:]
                self.extra_chars -= 1
            else:
                break

        self._raw(txt)