# -*- coding: utf-8 -*-
from lxml import objectify


def plot_state_border(plt, color='0.8'):
    path = 'states.xml'
    parsed = objectify.parse(open(path))
    root = parsed.getroot()
    points = []
    segments = []
    for state in root.state:
        for e in state.getchildren():
            points.append((e.get('lat'), e.get('lng')))

        for i in range(len(points) - 1):
            if i < len(points):
                segments.append((points[i], points[i + 1]))

        for (lat1, lon1), (lat2, lon2) in segments:
            plt.plot([lon1, lon2], [lat1, lat2], color=color)
        points = []
        segments = []
