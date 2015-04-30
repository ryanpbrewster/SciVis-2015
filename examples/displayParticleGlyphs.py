# Look at sdf_example.py in the examples/ directory for more detail

from sdfpy import load_sdf
from thingking import loadtxt
import vtk
import numpy as np

# Load N-body particles from a = 1.0 dataset. Particles have positions with
# units of proper kpc, and velocities with units of km/s. 
particles = load_sdf("../data/ds14_scivis_0128_e4_dt04_1.0000")
xx, yy, zz = particles['x'], particles['y'], particles['z']

n = 5000
pts = [ (xx[i], yy[i], zz[i]) for i in range(n) ]

points = vtk.vtkPoints()
for pt in pts:
    points.InsertNextPoint(pt)

grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(points)

sphere = vtk.vtkSphereSource()
sphere.SetRadius(50.0)

glyph = vtk.vtkGlyph3D()
glyph.SetInput(grid)
glyph.SetSource(sphere.GetOutput())

mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(glyph.GetOutput())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

ren = vtk.vtkRenderer()
ren.AddActor(actor)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Start()
