#!/usr/local/bin/python2.7
import bpy

# clean scene
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# add sphere in the center
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, location=(0,0,0))
sphere = bpy.data.objects['Icosphere']

# make sphere active and change to smooth shading
bpy.context.scene.objects.active = sphere
bpy.ops.object.shade_smooth()

# define origin empty object
bpy.ops.object.empty_add(location=(0,0,0))
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
bpy.data.scenes['Scene'].render.filepath = '/Users/carlovespa/Documents/Projects/Ball-Tracking/image.png'
bpy.data.scenes['Scene'].render.resolution_x = 210
bpy.data.scenes['Scene'].render.resolution_y = 210
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.alpha_mode = 'TRANSPARENT' # credit: http://blender.stackexchange.com/a/1307

bpy.ops.render.render(write_still = True)