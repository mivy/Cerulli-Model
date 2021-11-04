
import bpy
import math
import numpy
import pandas as pd


ce_constant = {
    'frame_interval' : 1,               #
    'frame_total' : 720,                 #
    'processes_nodes' : 4,             # 4, 40
    'samples' : 288,                    # 250, 250
    }   


def build_mat():
    bpy.data.materials.new("positive_mat")
    bpy.data.materials.new("negative_mat")
    bpy.data.materials.new("unsigned_mat")


def init_dataset_bpy(dataset_main, core):
    for iterator_i in range(core):
        
        v_loc = dataset_main[iterator_i]['location']
        v_loc = ''.join(v_loc.split('['))
        v_loc = ''.join(v_loc.split(']'))
        v_loc = v_loc.split(', ')
        
        v_loc[0] = float(v_loc[0])
        v_loc[1] = float(v_loc[1])
        v_loc[2] = float(v_loc[2])
        
        rad = dataset_main[iterator_i]['radius']
        
        bpy.ops.mesh.primitive_ico_sphere_add(location=(0,0,0), subdivisions=(1), radius=(rad), align='WORLD') # make obj
        bpy.ops.transform.translate(value=(v_loc))
        
        v_type = dataset_main[iterator_i]['type']
        bpy.context.object.data.materials.append(bpy.data.materials[f'{v_type}_mat']) # assign mat
        
        v_name = dataset_main[iterator_i]['name'] # generate name information
        bpy.context.object.name = v_name
        
        bpy.context.scene.objects[f'{v_name}'].keyframe_insert(data_path="location") # key
        
        if iterator_i == 5000:
            bpy.context.view_layer.update()
            print("...")
        
        


def anim_dataset_keyframes(dataset_main, core, frame):
    for iterator_n in range(core):

        blue_ct = core * (frame - 1) + iterator_n
        if frame == dataset_main[blue_ct]['frame']:
        
            v_name = dataset_main[blue_ct]['name']
            bpy.context.scene.objects[f'{v_name}'].select_set(state=True)
            
            v_vel = dataset_main[blue_ct]['velocity']
            v_vel = ''.join(v_vel.split('['))
            v_vel = ''.join(v_vel.split(']'))
            v_vel = v_vel.split(', ')
            
            v_vel[0] = float(v_vel[0])
            v_vel[1] = float(v_vel[1])
            v_vel[2] = float(v_vel[2])
            
            bpy.ops.transform.translate(value=v_vel)
            bpy.context.scene.objects[f'{v_name}'].keyframe_insert(data_path="location")
            
            #bpy.ops.transform.rotate(value=10)
            #bpy.context.scene.objects[f'{v_name}'].keyframe_insert(data_path="rotation")
            
            bpy.ops.object.select_all(action='DESELECT')
            
            if iterator_n == 5000:
                bpy.context.view_layer.update()
                print("...")
            
        else:
            continue
        
     
            
# ["frame", "name", "type", "mass", "radius", "location", "velocity"]       
if __name__ == "__main__":
    
    print("reading csv...")

    df = pd.read_csv('C:\data.csv', sep=",", header=[0], )
    dataset_main = df.to_dict('index')

    
    print("csv sucessfully cached")

    sample_sum = (ce_constant['processes_nodes'] * ce_constant['samples']) #- 1

    print("initializing buildset into bpy...")
    
    build_mat()
    
    init_dataset_bpy(dataset_main, sample_sum)

    print("bpy buildset done")
    print("building keyframes...")
        
    for frame in range(1, ce_constant['frame_total'], ce_constant['frame_interval']):
        
        bpy.context.scene.frame_set(frame)
        
        anim_dataset_keyframes(dataset_main, sample_sum, frame)
        
        print(f"done with frame {frame}")
        
     
    print("all keyframes complete")
