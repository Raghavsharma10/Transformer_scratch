def define_format(self, plotStyle, plotSize):

        #Default sizes for computer
        sizing_dict = {}
        sizing_dict['figure.figsize'] = (14, 8)
        sizing_dict['legend.fontsize'] = 15
        sizing_dict['axes.labelsize'] = 20
        sizing_dict['axes.titlesize'] = 24
        sizing_dict['xtick.labelsize'] = 14
        sizing_dict['ytick.labelsize'] = 14
        
        self.colorVector = {
        'iron':'#4c4c4c',
        'silver':'#cccccc',                  
        'dark blue':'#0072B2',
        'green':'#009E73', 
        'orangish':'#D55E00',
        'pink':'#CC79A7',
        'yellow':'#F0E442',
        'cyan':'#56B4E9',
        'olive':'#bcbd22',
        'grey':'#7f7f7f',
        'skin':'#FFB5B8'}

        #sizing_dict['text.usetex'] = True
        
        #--Update the colors/format
        if plotStyle == None:
            self.ColorVector = [None, None, None]
        
        elif plotStyle == 'dark':
            plt.style.use('dark_background')

        elif plotStyle == 'night':
            
            plt.style.use('seaborn-colorblind')
            
            iron_color = '#4c4c4c' #Iron: (76 76 76)
            silver_color = '#cccccc' #Silver: (204 204 204) 
            sizing_dict['axes.facecolor']   = iron_color
            sizing_dict['figure.facecolor'] = iron_color
            sizing_dict['axes.edgecolor']   = silver_color
            sizing_dict['text.color']       = silver_color
            sizing_dict['axes.labelcolor']  = silver_color
            sizing_dict['xtick.color']      = silver_color
            sizing_dict['ytick.color']      = silver_color
            sizing_dict['axes.edgecolor']   = silver_color

            
            #'plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))'
            #This should be the set up for the cycler we just need to add the colors
            #axes.prop_cycle : cycler('color', 'bgrcmyk')

        elif plotStyle == 'colorblind':
            plt.style.use('seaborn-colorblind')

        else:
            plt.style.use(plotStyle)
        
        #--Load particular configuration for this plot
        if plotSize == 'medium':            
            rcParams.update(sizing_dict)
        
        elif type(plotSize) is dict:
            sizing_dict.update(plotSize)
            rcParams.update(sizing_dict)

        '''
        Seaborn color blind
        #0072B2 dark blue
        #009E73 green 
        #D55E00 orangish
        #CC79A7 pink
        #F0E442 yellow
        #56B4E9 cyan
        #bcbd22 olive #adicional
        #7f7f7f grey
        #FFB5B8 skin
        '''
 
    
        '''
        Matplotlib default palete
        #17becf dark blue
        #bcbd22 orange
        #2ca02c green
        #e377c2 red
        #8c564b purple
        #9467bd brown
        #d62728 pink
        #7f7f7f grey
        #ff7f0e olive
        #1f77b4 cyan
        '''


  
        '''
        --These are matplotlib styles
        seaborn-darkgrid
        seaborn-notebook
        classic
        seaborn-ticks
        grayscale
        bmh
        seaborn-talk
        dark_background
        ggplot
        fivethirtyeight
        seaborn-colorblind
        seaborn-deep
        seaborn-whitegrid
        seaborn-bright
        seaborn-poster
        seaborn-muted
        seaborn-paper
        seaborn-white
        seaborn-pastel
        seaborn-dark
        seaborn
        seaborn-dark-palette
        '''