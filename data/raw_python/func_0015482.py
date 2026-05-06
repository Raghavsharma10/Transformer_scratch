def cat_proximity():
    """Cat proximity graph from http://xkcd.com/231/"""
    chart = SimpleLineChart(int(settings.width * 1.5), settings.height)
    chart.set_legend(['INTELLIGENCE', 'INSANITY OF STATEMENTS'])

    # intelligence
    data_index = chart.add_data([100. / y for y in range(1, 15)])

    # insanity of statements
    chart.add_data([100. - 100 / y for y in range(1, 15)])

    # line colours
    chart.set_colours(['208020', '202080'])

    # "Near" and "Far" labels, they are placed automatically at either ends.
    near_far_axis_index = chart.set_axis_labels(Axis.BOTTOM, ['FAR', 'NEAR'])

    # "Human Proximity to cat" label. Aligned to the center.
    index = chart.set_axis_labels(Axis.BOTTOM, ['HUMAN PROXIMITY TO CAT'])
    chart.set_axis_style(index, '202020', font_size=10, alignment=0)
    chart.set_axis_positions(index, [50])

    chart.download('label-cat-proximity.png')