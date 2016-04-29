#!/usr/local/bin/python3
import bpy
import random
import math
import os

# create rendering output directory, if it doesn't exist
spheresDir = os.path.join(os.getcwd(), './spheres')
if not os.path.exists(spheresDir):
     os.makedirs(spheresDir)

# clean scene
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# add sphere in the center
sphere = bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, location=(0,0,0))
sphere = bpy.data.objects['Icosphere']

# make sphere active and change to smooth shading
bpy.context.scene.objects.active = sphere
bpy.ops.object.shade_smooth()

# add origin empty object
origin = bpy.ops.object.empty_add(location=(0,0,0))
origin = bpy.data.objects['Empty']

# make camera active
camera = bpy.data.objects['Camera']
bpy.context.scene.objects.active = camera

# point camera to origin
# credit: http://blender.stackexchange.com/a/53
bpy.ops.object.constraint_add(type='TRACK_TO')
camera.constraints['Track To'].target = origin
camera.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
camera.constraints['Track To'].up_axis = 'UP_Y'

# position camera
camera.location = (2.5,0,0)

# render settings
scene = bpy.data.scenes['Scene']
scene.render.resolution_x = 210
scene.render.resolution_y = 210
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT' # credit: http://blender.stackexchange.com/a/1307

# enable environment lighting
bpy.data.worlds['World'].light_settings.use_environment_light = True

# get lights
light = bpy.data.objects['Lamp']

for x in range(10000):
    # position light randomly around center
    light_dist = random.random() * 5 + 5 # light distance in range [5, 10]
    light_longitude = random.random() * 2 * math.pi
    light_latitude = random.random() * 2 * math.pi
    light.location = ((math.sin(light_longitude) * light_dist, \
                       math.cos(light_longitude) * light_dist, \
                       math.sin(light_latitude) * light_dist))

    ambient = random.gauss(0.125, 0.125)
    if ambient < 0.0:
        ambient = 0.0
    if ambient > 1.0:
        ambient = 1.0

    bpy.data.worlds['World'].light_settings.environment_energy = ambient

    # save rendering in the 'renders' folder of the current working directory
    scene.render.filepath = os.path.join(spheresDir, './image%d.png' % x)
    bpy.ops.render.render(write_still = True)

