def barcode(self, data, format, characters='off', height=48, width='small', parentheses='on', ratio='3:1', equalize='off', rss_symbol='rss14std', horiz_char_rss=2):
        '''Print a standard barcode in the specified format
        
        Args:
            data: the barcode data
            format: the barcode type you want. Choose between code39, itf, ean8/upca, upce, codabar, code128, gs1-128, rss
            characters: Whether you want characters below the bar code. 'off' or 'on'
            height: Height, in dots.
            width: width of barcode. Choose 'xsmall' 'small' 'medium' 'large'
            parentheses: Parentheses deletion on or off. 'on' or 'off' Only matters with GS1-128
            ratio: ratio between thick and thin bars. Choose '3:1', '2.5:1', and '2:1'
            equalize: equalize bar lengths, choose 'off' or 'on'
            rss_symbol: rss symbols model, choose from 'rss14std', 'rss14trun', 'rss14stacked', 'rss14stackedomni', 'rsslimited', 'rssexpandedstd', 'rssexpandedstacked'
            horiz_char_rss: for rss expanded stacked, specify the number of horizontal characters, must be an even number b/w 2 and 20.
        '''
        
        barcodes = {'code39': '0',
                    'itf': '1',
                    'ean8/upca': '5',
                    'upce': '6',
                    'codabar': '9',
                    'code128': 'a',
                    'gs1-128': 'b',
                    'rss': 'c'}
        
        widths = {'xsmall': '0',
                  'small': '1',
                  'medium': '2',
                  'large': '3'}
        
        ratios = {'3:1': '0',
                  '2.5:1': '1',
                  '2:1': '2'}
        
        rss_symbols = {'rss14std': '0',
                       'rss14trun': '1',
                       'rss14stacked': '2',
                       'rss14stackedomni' : '3',
                       'rsslimited': '4',
                       'rssexpandedstd': '5',
                       'rssexpandedstacked': '6'
                       }
        
        character_choices = {'off': '0',
                      'on' : '1'}
        parentheses_choices = {'off':'1',
                               'on': '0'}
        equalize_choices = {'off': '0',
                            'on': '1'}
        
        sendstr = ''
        n2 = height/256
        n1 = height%256
        if format in barcodes and width in widths and ratio in ratios and characters in character_choices and rss_symbol in rss_symbols:
            sendstr += (chr(27)+'i'+'t'+barcodes[format]+'s'+'p'+'r'+character_choices[characters]+'u'+'x'+'y'+'h' + chr(n1) + chr(n2) +
                        'w'+widths[width]+'e'+parentheses_choices[parentheses]+'o'+rss_symbols[rss_symbol]+'c'+chr(horiz_char_rss)+'z'+ratios[ratio]+'f'+equalize_choices[equalize]
                        + 'b' + data + chr(92))
            if format in ['code128', 'gs1-128']:
                sendstr += chr(92)+ chr(92)
            self.send(sendstr)
        else:
            raise RuntimeError('Invalid parameters')