#!/usr/local/bin/python3
import bpy
import random
import math

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

# get light
light = bpy.data.objects['Lamp']

for x in range(10):
    # position light randomly around center
    light_dist = random.random() * 5 + 5 # light distance in range [5, 10]
    light.location = ((math.sin(random.random() * 2 * math.pi) * light_dist, \
                       math.sin(random.random() * 2 * math.pi) * light_dist, \
                       math.sin(random.random() * 2 * math.pi) * light_dist))
    
    scene.render.filepath = './renders/image%d.png' % x
    bpy.ops.render.render(write_still = True)