from itertools import combinations
import pyqtgraph as pg

def create_color_maps():
    cmap_names = ["CET-L13", "CET-L14", "CET-L15"]
    cmap_combo = combinations(cmap_names, 2)
    cmap_label1 = ["red", "green", "blue"]
    cmap_label2 = ["yellow", "magenta", "cyan"]
    cmap_dict = {}
    for i, name in zip(cmap_names, cmap_label1):
        cmap_dict[name] = pg.colormap.get(i).getLookupTable(alpha=True)

    for i, name in zip(cmap_combo, cmap_label2):
        cmap_dict[name] = pg.colormap.get(i[0]).getLookupTable(alpha=True) + pg.colormap.get(i[1]).getLookupTable(
            alpha=True
        )
        cmap_dict[name][:, 3] = 255

        grey = (
            pg.colormap.get("CET-L13").getLookupTable(alpha=True)
            + pg.colormap.get("CET-L14").getLookupTable(alpha=True)
            + pg.colormap.get("CET-L15").getLookupTable(alpha=True)
        )

        grey[:, 3] = 255
        cmap_dict["grey"] = grey

        return cmap_dict