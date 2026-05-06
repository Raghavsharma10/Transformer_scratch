def add_line(preso, x1, y1, x2, y2, width="3pt", color="red"):
    """
    Arrow pointing up to right:

    context.xml:

office:automatic-styles/
    <style:style style:name="gr1" style:family="graphic" style:parent-style-name="objectwithoutfill">
    <style:graphic-properties
      draw:marker-end="Arrow"
      draw:marker-end-width="0.3cm"
      draw:fill="none"
      draw:textarea-vertical-align="middle"/>
    </style:style>

    3pt width color red
<style:style style:name="gr2" style:family="graphic" style:parent-style-name="objectwithoutfill">
    <style:graphic-properties
      svg:stroke-width="0.106cm"
      svg:stroke-color="#ed1c24" 
      draw:marker-start-width="0.359cm"
      draw:marker-end="Arrow"
      draw:marker-end-width="0.459cm"
      draw:fill="none"
      draw:textarea-vertical-align="middle"
      fo:padding-top="0.178cm"
      fo:padding-bottom="0.178cm"
      fo:padding-left="0.303cm"
      fo:padding-right="0.303cm"/>
    </style:style>




    ...

office:presentation/draw:page

    <draw:line draw:style-name="gr1" draw:text-style-name="P2" draw:layer="layout" svg:x1="6.35cm" svg:y1="10.16cm" svg:x2="10.668cm" svg:y2="5.842cm"><text:p/></draw:line>
    """
    marker_end_ratio = .459 / 3  # .459cm/3pt
    marker_start_ratio = .359 / 3  # .359cm/3pt
    stroke_ratio = .106 / 3  # .106cm/3pt

    w = float(width[0:width.index("pt")])
    sw = w * stroke_ratio
    mew = w * marker_end_ratio
    msw = w * marker_start_ratio
    attribs = {
        "svg:stroke-width": "{}cm".format(sw),
        "svg:stroke-color": color,  # "#ed1c24",
        "draw:marker-start-width": "{}cm".format(msw),
        "draw:marker-end": "Arrow",
        "draw:marker-end-width": "{}cm".format(mew),
        "draw:fill": "none",
        "draw:textarea-vertical-align": "middle",
    }
    style = LineStyle(**attribs)
    # node = style.style_node()
    preso.add_style(style)
    line_attrib = {
        "draw:style-name": style.name,
        "draw:layer": "layout",
        "svg:x1": x1,
        "svg:y1": y1,
        "svg:x2": x2,
        "svg:y2": y2,
    }
    line_node = el("draw:line", attrib=line_attrib)
    preso.slides[-1]._page.append(line_node)