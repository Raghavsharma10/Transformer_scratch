def receipt(self,xml):
        """
        Prints an xml based receipt definition
        """

        def strclean(string):
            if not string:
                string = ''
            string = string.strip()
            string = re.sub('\s+',' ',string)
            return string

        def format_value(value, decimals=3, width=0, decimals_separator='.', thousands_separator=',', autoint=False, symbol='', position='after'):
            decimals = max(0,int(decimals))
            width    = max(0,int(width))
            value    = float(value)

            if autoint and math.floor(value) == value:
                decimals = 0
            if width == 0:
                width = ''

            if thousands_separator:
                formatstr = "{:"+str(width)+",."+str(decimals)+"f}"
            else:
                formatstr = "{:"+str(width)+"."+str(decimals)+"f}"


            ret = formatstr.format(value)
            ret = ret.replace(',','COMMA')
            ret = ret.replace('.','DOT')
            ret = ret.replace('COMMA',thousands_separator)
            ret = ret.replace('DOT',decimals_separator)

            if symbol:
                if position == 'after':
                    ret = ret + symbol
                else:
                    ret = symbol + ret
            return ret

        def print_elem(stylestack, serializer, elem, indent=0):

            elem_styles = {
                'h1': {'bold': 'on', 'size':'double'},
                'h2': {'size':'double'},
                'h3': {'bold': 'on', 'size':'double-height'},
                'h4': {'size': 'double-height'},
                'h5': {'bold': 'on'},
                'em': {'font': 'b'},
                'b':  {'bold': 'on'},
            }

            stylestack.push()
            if elem.tag in elem_styles:
                stylestack.set(elem_styles[elem.tag])
            stylestack.set(elem.attrib)

            if elem.tag in ('p','div','section','article','receipt','header','footer','li','h1','h2','h3','h4','h5'):
                serializer.start_block(stylestack)
                serializer.text(elem.text)
                for child in elem:
                    print_elem(stylestack,serializer,child)
                    serializer.start_inline(stylestack)
                    serializer.text(child.tail)
                    serializer.end_entity()
                serializer.end_entity()

            elif elem.tag in ('span','em','b','left','right'):
                serializer.start_inline(stylestack)
                serializer.text(elem.text)
                for child in elem:
                    print_elem(stylestack,serializer,child)
                    serializer.start_inline(stylestack)
                    serializer.text(child.tail)
                    serializer.end_entity()
                serializer.end_entity()

            elif elem.tag == 'value':
                serializer.start_inline(stylestack)
                serializer.pre(format_value( 
                                              elem.text,
                                              decimals=stylestack.get('value-decimals'),
                                              width=stylestack.get('value-width'),
                                              decimals_separator=stylestack.get('value-decimals-separator'),
                                              thousands_separator=stylestack.get('value-thousands-separator'),
                                              autoint=(stylestack.get('value-autoint') == 'on'),
                                              symbol=stylestack.get('value-symbol'),
                                              position=stylestack.get('value-symbol-position') 
                                            ))
                serializer.end_entity()

            elif elem.tag == 'line':
                width = stylestack.get('width')
                if stylestack.get('size') in ('double', 'double-width'):
                    width = width / 2

                lineserializer = XmlLineSerializer(stylestack.get('indent')+indent,stylestack.get('tabwidth'),width,stylestack.get('line-ratio'))
                serializer.start_block(stylestack)
                for child in elem:
                    if child.tag == 'left':
                        print_elem(stylestack,lineserializer,child,indent=indent)
                    elif child.tag == 'right':
                        lineserializer.start_right()
                        print_elem(stylestack,lineserializer,child,indent=indent)
                serializer.pre(lineserializer.get_line())
                serializer.end_entity()

            elif elem.tag == 'ul':
                serializer.start_block(stylestack)
                bullet = stylestack.get('bullet')
                for child in elem:
                    if child.tag == 'li':
                        serializer.style(stylestack)
                        serializer.raw(' ' * indent * stylestack.get('tabwidth') + bullet)
                    print_elem(stylestack,serializer,child,indent=indent+1)
                serializer.end_entity()

            elif elem.tag == 'ol':
                cwidth = len(str(len(elem))) + 2
                i = 1
                serializer.start_block(stylestack)
                for child in elem:
                    if child.tag == 'li':
                        serializer.style(stylestack)
                        serializer.raw(' ' * indent * stylestack.get('tabwidth') + ' ' + (str(i)+')').ljust(cwidth))
                        i = i + 1
                    print_elem(stylestack,serializer,child,indent=indent+1)
                serializer.end_entity()

            elif elem.tag == 'pre':
                serializer.start_block(stylestack)
                serializer.pre(elem.text)
                serializer.end_entity()

            elif elem.tag == 'hr':
                width = stylestack.get('width')
                if stylestack.get('size') in ('double', 'double-width'):
                    width = width / 2
                serializer.start_block(stylestack)
                serializer.text('-'*width)
                serializer.end_entity()

            elif elem.tag == 'br':
                serializer.linebreak()

            elif elem.tag == 'img':
                if 'src' in elem.attrib and 'data:' in elem.attrib['src']:
                    self.print_base64_image(elem.attrib['src'])

            elif elem.tag == 'barcode' and 'encoding' in elem.attrib:
                serializer.start_block(stylestack)
                self.barcode(strclean(elem.text),elem.attrib['encoding'])
                serializer.end_entity()

            elif elem.tag == 'cut':
                self.cut()
            elif elem.tag == 'partialcut':
                self.cut(mode='part')
            elif elem.tag == 'cashdraw':
                self.cashdraw(2)
                self.cashdraw(5)

            stylestack.pop()

        try:
            stylestack      = StyleStack() 
            serializer      = XmlSerializer(self)
            root            = ET.fromstring(xml.encode('utf-8'))
            if 'sheet' in root.attrib and root.attrib['sheet'] == 'slip':
                self._raw(SHEET_SLIP_MODE)
                self.slip_sheet_mode = True
            else:
                self._raw(SHEET_ROLL_MODE)

            self._raw(stylestack.to_escpos())

            print_elem(stylestack,serializer,root)

            if 'open-cashdrawer' in root.attrib and root.attrib['open-cashdrawer'] == 'true':
                self.cashdraw(2)
                self.cashdraw(5)
            if not 'cut' in root.attrib or root.attrib['cut'] == 'true' :
                if self.slip_sheet_mode:
                    self._raw(CTL_FF)
                else:
                    self.cut()

        except Exception as e:
            errmsg = str(e)+'\n'+'-'*48+'\n'+traceback.format_exc() + '-'*48+'\n'
            self.text(errmsg)
            self.cut()

            raise e