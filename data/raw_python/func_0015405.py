def house_explosions():
    """
    Data from http://indexed.blogspot.com/2007/12/meltdown-indeed.html
    """
    chart = PieChart2D(int(settings.width * 1.7), settings.height)
    chart.add_data([10, 10, 30, 200])
    chart.set_pie_labels([
        'Budding Chemists',
        'Propane issues',
        'Meth Labs',
        'Attempts to escape morgage',
        ])
    chart.download('pie-house-explosions.png')