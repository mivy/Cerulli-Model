# Project Cerulli by Mivy

import pandas as pd
import math
import numpy as np
import multiprocessing as mp

from numpy.random.mtrand import f


ce_constant = {
    'ngc' : float(6.674080000e-11),      # 6.67408e-11  
    'e' : 100,                          #
    'frame_interval' : 1,               #
    'frame_total' : 100,                 #
    'time' : 1,                         #
    'processes_nodes' : 40,             # 4, 40
    'samples' : 250,                    # 250, 250
    }                        # samples: # 1_000, 10_000
aim = {}


def loopContainer(queue, i):
    ret = queue.get()
    for k in range(ce_constant['samples']): # total indicies counted here
        set_list_index = nQueue(i, k) # counter for the name trait
        initialize_normal = np.random.default_rng().standard_normal(4) # 
        initialize_kmass = abs(initialize_normal[3]) # base value for mass distribution
        initialize_random = np.random.randint(99)
        set_frame = 0
        set_name = f'n_{set_list_index}'
        set_type = iType(initialize_kmass, initialize_random)
        set_mass = iMass(set_type, initialize_kmass)
        set_visual_radius = initialize_kmass * .01 # object volume (see: bpy)
        set_location = [initialize_normal[0], initialize_normal[1], initialize_normal[2]]
        set_velocity = [0,0,0]
        initial_set = [set_frame, set_name, set_type, set_mass, set_visual_radius, set_location, set_velocity]
        ret[set_list_index] = initial_set # return thread result
        # print(initial_set) # test print [working]
    queue.put(ret) # return process result


def iType(i, ii): # assign types
    if i > 3: #
        m = 'unsigned'
    elif ii <= 80: # 15 universally
        m = 'positive'
    else:
        m = 'negative'
    return m


def iMass(i, ii): # assign mass
    if i == 'unsigned':
        ii = (ii + 1) * ((ii + 4) / 2) # about 10 sm
    elif i == 'negative':
        ii = ii * (-1) # reductive
    ii = ii * 100000
    return ii


def coreFunction(shared_list, history, core, frame):
    #aim = queue.get()
    #print(history[3])

    for focus in range(core[0], core[1]): # total indicies counted here
        focus_object = history[focus] # and call its values
        acceleration_vector = [0,0,0] # formatting (check function)
        for target in range(len(history)): # compares to all other indices
            if focus == target: # but not itself
                continue
            target_object = history[target]
            result_distance = nPythagorean(focus_object, target_object)
            acceleration_vector = nSumVector(focus_object, target_object, result_distance, acceleration_vector) # find acceleration between focus and n targets

            # add additional interactions here
        velocity_vector = focus_object[6]
        location = focus_object[5]
        
        #print(velocity_vector[2]) # working

        velocity_delta = nDeltaTime(acceleration_vector, velocity_vector)
        
        location_delta = nLocationUpdate(location, velocity_delta)
        
    
        

        set_frame = frame
        set_name = focus_object[1]
        set_type = focus_object[2]
        set_mass = focus_object[3]
        set_visual_radius = focus_object[4]
        set_location = location_delta
        set_velocity = velocity_delta
        aim[focus] = [set_frame, set_name, set_type, set_mass, set_visual_radius, set_location, set_velocity]
        shared_list.append(aim[focus])
    #queue.put(aim)


def nQueue(i, ii): # assign iterations
    init_i = ii * (ce_constant['processes_nodes']) + i
    return init_i


def nPythagorean(f, t):
    p_distance = math.sqrt(((t[5][0] - f[5][0])**2) + ((t[5][1] - f[5][1])**2) + ((t[5][2] - f[5][2])**2))
    return p_distance


def nSumVector(f, t, ret_distance, acceleration_vector_sum):
    acceleration_vector = [0,0,0]
    for i in range(3): # where range is dimensions of gravity # ret_distance**3
        newton = (-1) * ((ce_constant['ngc']) * (t[3]) * (f[5][i] - t[5][i])) / (ret_distance**3) # VARIABLE E ...  + (.0000000000000002**2))**(3/2)
        acceleration_vector[i] = newton
    new_vector = [acceleration_vector[0] + acceleration_vector_sum[0], 
    acceleration_vector[1] + acceleration_vector_sum[1], 
    acceleration_vector[2] + acceleration_vector_sum[2]]
    return new_vector


def nDeltaTime(acceleration_vector, velocity_vector):
    delta_v = [0,0,0]
    for i in range(3):
        delta_v[i] =  ((ce_constant['time']**2) * acceleration_vector[i]) + (velocity_vector[i] * ce_constant['time'])
    return delta_v


def nLocationUpdate(location, velocity):
    new_location = [0,0,0]
    for i in range(3):
        new_location[i] = location[i] + velocity[i]
    return new_location


#def qPythagorean(t, f):
#    if ttl_r > abs(t[3][0] - f[3][0]) and ttl_r > abs(t[3][1] - f[3][1]) and ttl_r > abs(t[3][2] - f[3][2]): 
#        close_enough = True
#    return close_enough


if __name__ == "__main__":
    queue = mp.Manager().Queue()
    ret = {}
    export_data_set = []
    updated_list_of_values = []

    processes_n = []
    queue.put(ret)
    for i in range(ce_constant['processes_nodes']):
        p = mp.Process(target=loopContainer, args=(queue, i,))
        p.start()
        processes_n.append(p)
    for p in processes_n:
        p.join()
    ret = queue.get()
    updated_list_of_values = list(ret.values())
    #print(history)
    #ret = list(ret.values()) # fixed when 'ret' is a list, otherwise convert the dict into list


    core = []
    processes_n = []
    #queue.put(aim)
    sample_sum = ce_constant['processes_nodes'] * ce_constant['samples']
    for i in range(ce_constant['processes_nodes']):
        i_process_sum = ce_constant['samples'] * i
        k_process_sum = ce_constant['samples'] * (i + 1)
        core.append([i_process_sum, k_process_sum])

    for frame in range(1, ce_constant['frame_total'], ce_constant['frame_interval']):
        shared_list = mp.Manager().list()
        for i in range(ce_constant['processes_nodes']):
            p = mp.Process(target=coreFunction, args=(shared_list, updated_list_of_values, core[i], frame,))
            p.start()
            processes_n.append(p)
        for p in processes_n:
            p.join()
        

        updated_list_of_values = list(shared_list)
        
        appendation = list(shared_list)
        export_data_set.extend(appendation)
        print("done with frame ", frame)
        
        print(updated_list_of_values[3]) # test print

    #export_data_set = list(export_data_set)
    #print(export_data_set)
    #export_data_set = list(export_data_set.items())
    #print(export_data_set) # test print
    pd_dataframe = pd.DataFrame(export_data_set, columns=["frame", "name", "type", "mass", "radius", "location", "velocity"])
    pd_dataframe.to_csv(r'C:\data.csv')
    print("end1")
    print("end2")
# breakpoint end   [set_frame, set_name, set_type, set_mass, set_visual_radius, set_location, set_velocity]
