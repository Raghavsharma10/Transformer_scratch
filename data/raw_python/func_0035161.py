def _my_eps_formatter(logodata, format, ordered_alphabets) :
    """ Generate a logo in Encapsulated Postscript (EPS)
    
    Modified from weblogo version 3.4 source code. 

    *ordered_alphabets* is a dictionary keyed by zero-indexed
    consecutive sites, with values giving order of characters
    from bottom to top.
    """
    substitutions = {}
    from_format =[
        "creation_date",    "logo_width",           "logo_height",      
        "lines_per_logo",   "line_width",           "line_height",
        "line_margin_right","line_margin_left",     "line_margin_bottom",
        "line_margin_top",  "title_height",         "xaxis_label_height",
        "creator_text",     "logo_title",           "logo_margin",
        "stroke_width",     "tic_length",           
        "stacks_per_line",  "stack_margin",
        "yaxis_label",      "yaxis_tic_interval",   "yaxis_minor_tic_interval",
        "xaxis_label",      "xaxis_tic_interval",   "number_interval",
        "fineprint",        "shrink_fraction",      "errorbar_fraction",
        "errorbar_width_fraction",
        "errorbar_gray",    "small_fontsize",       "fontsize",
        "title_fontsize",   "number_fontsize",      "text_font",
        "logo_font",        "title_font",          
        "logo_label",       "yaxis_scale",          "end_type",
        "debug",            "show_title",           "show_xaxis",
        "show_xaxis_label", "show_yaxis",           "show_yaxis_label",
        "show_boxes",       "show_errorbars",       "show_fineprint",
        "rotate_numbers",   "show_ends",            "stack_height",
        "stack_width"
        ]
   
    for s in from_format :
        substitutions[s] = getattr(format,s)

    substitutions["shrink"] = str(format.show_boxes).lower()


    # --------- COLORS --------------
    def format_color(color):
        return  " ".join( ("[",str(color.red) , str(color.green), 
            str(color.blue), "]"))  

    substitutions["default_color"] = format_color(format.default_color)

    colors = []  
    if hasattr(format.color_scheme, 'rules'):
        grouplist = format.color_scheme.rules
    else:
        # this line needed for weblogo 3.4
        grouplist = format.color_scheme.groups
    for group in grouplist:
        cf = format_color(group.color)
        for s in group.symbols :
            colors.append( "  ("+s+") " + cf )
    substitutions["color_dict"] = "\n".join(colors)
        
    data = []
    
    # Unit conversion. 'None' for probability units
    conv_factor = None #JDB
    #JDB conv_factor = std_units[format.unit_name]
    
    data.append("StartLine")

    seq_from = format.logo_start- format.first_index
    seq_to = format.logo_end - format.first_index +1

    # seq_index : zero based index into sequence data
    # logo_index : User visible coordinate, first_index based
    # stack_index : zero based index of visible stacks
    for seq_index in range(seq_from, seq_to) :
        logo_index = seq_index + format.first_index 
        stack_index = seq_index - seq_from
        
        if stack_index!=0 and (stack_index % format.stacks_per_line) ==0 :
            data.append("")
            data.append("EndLine")
            data.append("StartLine")
            data.append("")
        
        data.append("(%s) StartStack" % format.annotate[seq_index] )

        if conv_factor: 
            stack_height = logodata.entropy[seq_index] * std_units[format.unit_name]
        else :
            stack_height = 1.0 # Probability

        # The following code modified by JDB to use ordered_alphabets
        # and also to replace the "blank" characters 'b' and 'B'
        # by spaces.
        s_d = dict(zip(logodata.alphabet, logodata.counts[seq_index]))
        s = []
        for aa in ordered_alphabets[seq_index]:
            if aa not in ['B', 'b']:
                s.append((s_d[aa], aa))
            else:
                s.append((s_d[aa], ' '))
#        s = [(s_d[aa], aa) for aa in ordered_alphabets[seq_index]]

        # Sort by frequency. If equal frequency then reverse alphabetic
        # (So sort reverse alphabetic first, then frequencty)
        # TODO: doublecheck this actual works
        #s = list(zip(logodata.counts[seq_index], logodata.alphabet))
        #s.sort(key= lambda x: x[1])
        #s.reverse()
        #s.sort(key= lambda x: x[0])
        #if not format.reverse_stacks: s.reverse()

        C = float(sum(logodata.counts[seq_index])) 
        if C > 0.0 :
            fraction_width = 1.0
            if format.scale_width :
                fraction_width = logodata.weight[seq_index] 
            # print(fraction_width, file=sys.stderr)
            for c in s:
                data.append(" %f %f (%s) ShowSymbol" % (fraction_width, c[0]*stack_height/C, c[1]) )

        # Draw error bar on top of logo. Replaced by DrawErrorbarFirst above.
        if logodata.entropy_interval is not None and conv_factor and C>0.0:

            low, high = logodata.entropy_interval[seq_index]
            center = logodata.entropy[seq_index]
            low *= conv_factor
            high *= conv_factor
            center *=conv_factor
            if high> format.yaxis_scale : high = format.yaxis_scale 

            down = (center - low) 
            up   = (high - center) 
            data.append(" %f %f DrawErrorbar" % (down, up) )
            
        data.append("EndStack")
        data.append("")
               
    data.append("EndLine")
    substitutions["logo_data"] = "\n".join(data)  


    # Create and output logo
    template = corebio.utils.resource_string( __name__, '_weblogo_template.eps', __file__).decode()
    logo = string.Template(template).substitute(substitutions)

    return logo.encode()