# Project Cerulli by Mivy

import bpy
import pandas as pd


ce_constant = {
    'frame_interval' : 1,               #
    'frame_total' : 40,                 #
    'processes_nodes' : 10,             # 4, 40
    'samples' : 100,                    # 250, 250
    }   


def build_mat():
    bpy.data.materials.new("positive_mat")
    #bpy.data.materials.new("negative_mat")

    
def init_dataset_bpy(dataset_main, core):
    for i in range(core): # ["name", "mass", "radius", "lx", "ly", "lz", "vx", "vy", "vz"]
        x = dataset_main[i]['lx']
        y = dataset_main[i]['ly']
        z = dataset_main[i]['lz']
        r = dataset_main[i]['radius']
        bpy.ops.mesh.primitive_ico_sphere_add(location=(x,y,z), subdivisions=(1), radius=(r), align='WORLD') # make obj
        bpy.context.object.data.materials.append(bpy.data.materials['positive_mat']) # assign mat
        name = dataset_main[i]['name'] # generate name information
        bpy.context.object.name = name
        bpy.context.scene.objects[f'{name}'].keyframe_insert(data_path="location") # key
        

def anim_dataset_keyframes(dataset_main, core, frame):
    for iterator_n in range(core): # ["frame", "name", "x", "y", "z"]
        ct = core * (frame - 1) + iterator_n
        if frame == dataset_main[ct]['frame']:
            name = dataset_main[ct]['name']
            bpy.context.scene.objects[f'{name}'].select_set(state=True)
            x = dataset_main[ct]['x']
            y = dataset_main[ct]['y']
            z = dataset_main[ct]['z']
            bpy.data.objects[f'{name}'].location = (x, y, z)
            bpy.context.scene.objects[f'{name}'].keyframe_insert(data_path="location")
            #bpy.ops.transform.rotate(value=10)
            #bpy.context.scene.objects[f'{v_name}'].keyframe_insert(data_path="rotation")
            bpy.context.scene.objects[f'{name}'].select_set(state=False)
            #bpy.ops.object.select_all(action='DESELECT')
        else:
            continue

            
if __name__ == "__main__":
    print("reading csv's...")
    sample_sum = (ce_constant['processes_nodes'] * ce_constant['samples']) #- 1
    df = pd.read_csv(r'C:\init1.csv', sep=",", header=[0], )
    dataset_init = df.to_dict('index')
    df = pd.read_csv(r'C:\anim1.csv', sep=",", header=[0], )
    dataset_anim = df.to_dict('index')
    print("csv's sucessfully cached")
    print("initializing bpy...")
    build_mat()
    init_dataset_bpy(dataset_init, sample_sum)
    print("bpy initialized")
    print("distributing keyframes...")
    for frame in range(2, ce_constant['frame_total']+1, ce_constant['frame_interval']):
        bpy.context.scene.frame_set(frame)
        anim_dataset_keyframes(dataset_anim, sample_sum, frame)
        print(f"done with frame {frame}")
    print("all keyframes complete")
