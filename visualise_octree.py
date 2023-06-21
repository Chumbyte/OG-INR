import os, sys, time
import numpy as np
import itertools

################################
## Plotting Tree
################################

import pyvista as pv
from pyvista import _vtk
def pvCube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0):
    src = _vtk.vtkCubeSource()
    src.SetCenter(center)
    src.SetXLength(x_length)
    src.SetYLength(y_length)
    src.SetZLength(z_length)
    src.Update()
    cube = pv.wrap(src.GetOutput())
    return cube


def vis_octree(mnfld_pnts, ot, labels, is2D=False):
    print_times = True
    if is2D:
        print_times = False
    print('Plotting')
    first_t0 = time.time()
    t0 = time.time()
    pv.global_theme.background = 'black'
    pv.global_theme.color = 'white'
    pv.global_theme.font.color = 'white'
    pv.global_theme.edge_color = 'white'
    pv.global_theme.outline_color = 'white'

    pcd = pv.PolyData(mnfld_pnts[:])
    pl = pv.Plotter(window_size=(1024, 768))
    # pl = pv.Plotter(window_size=(1024, 768), off_screen=True)
    # pl.enable_depth_peeling(10)
    if print_times: print('\tStarted Plotter ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

    vis_sc = 0.9 # scale down box length for visualisation
    outside = pv.MultiBlock()
    surf = pv.MultiBlock()
    inside = pv.MultiBlock()

    for node in ot.surfaceLeaves:
        l = node.length
        x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
        pv_cube = pvCube(center=(x+l/2, y+l/2, z+l/2), 
                x_length = vis_sc * l, 
                y_length = vis_sc * l, 
                z_length = vis_sc * l)
        surf.append(pv_cube)
    if print_times: print('\tSurface MultiBlock made ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

    for i, node in enumerate(ot.nonSurfaceLeaves):
        l = node.length
        x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
        pv_cube = pvCube(center=(x+l/2, y+l/2, z+l/2), 
                x_length = vis_sc * l, 
                y_length = vis_sc * l, 
                z_length = vis_sc * l)
        if labels[i]:
            inside.append(pv_cube)
        else:
            outside.append(pv_cube)
    if print_times: print('\tNon-Surf MultiBlocks made ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

    num_outside, num_inside, num_surf = len(outside), len(inside), len(surf)
    print(f"\t# outside: {num_outside}, # inside: {num_inside}, # surface: {num_surf}")

    outside = outside.combine(merge_points=True)
    surf = surf.combine(merge_points=True)
    inside = inside.combine(merge_points=True)
    if print_times: print('\tMultiBlocks combined ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

    if num_surf > 0:
        pl.add_mesh(surf, color='yellow', opacity=1.0, style='surface', line_width=0.01, name='surface')
    if num_inside > 0:
        pl.add_mesh(inside, color='blue', opacity=1.0, style='surface', line_width=0.01, name='inside')
    if num_outside > 0:
        pl.add_mesh(outside, color='green', opacity=1.0 if is2D else 0.01, style='surface' if is2D else 'wireframe', line_width=0.01, name='outside')
    
    pl.add_points(pcd.points, color='white', point_size=1)
    if print_times: print('\tAdded combined meshes and points ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

    class WidgetCallback:
        """Helper callback to keep a reference to the actor being modified."""

        def __init__(self):
            self.args = {
                0 : {'opacity': 1.0 if is2D else 0.01, 'style': 'surface' if is2D else 'wireframe'},
                1 : {'opacity': 1.0, 'style': 'surface'},
                2 : {'opacity': 1.0, 'style': 'surface'},
            }

        def __call__(self, leaf_type_num, attr, val):
            if attr == 'style':
                val = 'surface' if val else 'wireframe'
            self.args[leaf_type_num][attr] = val
            self.update()
        
        def update(self):
            if num_surf > 0:
                pl.add_mesh(surf, color='yellow', opacity=self.args[1]['opacity'],
                            style=self.args[1]['style'], line_width=0.01, name='surface')
            if num_inside > 0:
                pl.add_mesh(inside, color='blue', opacity=self.args[2]['opacity'],
                            style=self.args[2]['style'], line_width=0.01, name='inside')
            if num_outside > 0:
                pl.add_mesh(outside, color='green', opacity=self.args[0]['opacity'],
                            style=self.args[0]['style'], line_width=0.01, name='outside')

    widgetCallback = WidgetCallback()

    pl.add_slider_widget(
        callback=lambda value: widgetCallback(0, 'opacity', float(value)),
        rng=[0, 1],
        value=1.0 if is2D else 0.01,
        title="Outside Opacity",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
    )

    pl.add_slider_widget(
        callback=lambda value: widgetCallback(1, 'opacity', float(value)),
        rng=[0, 1],
        value=1.0,
        pointa=(0.35, 0.1),
        pointb=(0.64, 0.1),
        title="Surface Opacity",
        style='modern',
    )

    pl.add_slider_widget(
        callback=lambda value: widgetCallback(2, 'opacity', float(value)),
        rng=[0, 1],
        value=1.0,
        title="Inside Opacity",
        pointa=(0.67, 0.1),
        pointb=(0.98, 0.1),
        style='modern',
    )

    pl.add_checkbox_button_widget(
        callback=lambda flag: widgetCallback(0, 'style', flag),
        value=True if is2D else False,
        position=(10, 200),
        # size=size,
        border_size=1,
        color_on='green',
        color_off='grey',
        background_color='grey',
    )

    pl.add_checkbox_button_widget(
        callback=lambda flag: widgetCallback(1, 'style', flag),
        value=True,
        position=(10, 300),
        # size=size,
        border_size=1,
        color_on='yellow',
        color_off='grey',
        background_color='grey',
    )

    pl.add_checkbox_button_widget(
        callback=lambda flag: widgetCallback(2, 'style', flag),
        value=True,
        position=(10, 400),
        # size=size,
        border_size=1,
        color_on='blue',
        color_off='grey',
        background_color='grey',
    )

    pl.add_axes()
    if print_times: print('\tAdded widgets and axes ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    
    if is2D:
        pl.camera.position = (0.1, 0.1, 1.0)
        pl.camera.tight()

    print('#### Overall Plot Time ({:.5f}s)'.format(time.time()-first_t0))
    pl.show()
    # pl.show(screenshot='screenshot.png')

    # # import pdb; pdb.set_trace()