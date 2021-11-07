# Project Cerulli by Mivy

import pandas as pd
import math
import numpy as np
import multiprocessing as mp


const = {
    'newton' : float(6.674080000e-11),  # 6.67408e-11  
    'e' : .11,                          #
    'frame_interval' : 1,               #
    'frame_total' : 60,                 #
    'time' : 1,                         #
    'processes_nodes' : 10,             #
    'samples' : 100,                    #
    'mass_p_law' : -4.5,                #
    'mass_scalar' : 100000,             #
    'location_scalar' : 3,              #
    'velocity_scalar' : 1,              #
    'radius_scalar' : .00001,           # includes radius for volume scatter shader //i.e. cosmic dust
    }


def index_counter(i, ii): # assign indicies for init and anim
    init_i = ii * (const['processes_nodes']) + i
    return init_i


def init_dataset(queue, i): # columns=["name", "type", "mass", "radius", "lx", "ly", "lz", deltax, deltay, deltaz]
    ret = queue.get()
    for k in range(const['samples']): # total indicies counted here
        list_index = index_counter(i, k) # counter for the name trait
        set_name = f'n_{list_index}'
        set_mass = mass_distribution()
        set_vrad = abs(set_mass) * const['radius_scalar']
        coord_i = np.random.default_rng().standard_normal() * const['location_scalar']
        coord_j = np.random.default_rng().standard_normal() * const['location_scalar']
        coord_k = np.random.default_rng().standard_normal() * const['location_scalar']
        delta_x = 0 # coord_i * const['velocity_scalar']
        delta_y = 0 # coord_j * const['velocity_scalar']
        delta_z = 0 # coord_k * const['velocity_scalar']
        initial_set = [set_name, set_mass, set_vrad, coord_i, coord_j, coord_k, delta_x, delta_y, delta_z]
        ret[list_index] = initial_set # return thread result
    queue.put(ret) # return process result


def mass_distribution(): # truncated power law for m
    x = abs(np.random.default_rng().standard_normal())
    t = 1 + 1 / x ** const['mass_p_law']
    t = t * const['mass_scalar']
    return t


def anim_dataset(queue, history, core, frame): # columns=["frame", "name", "vx", "vy", "vz"] previous_set, anim_set, 
    anim_set = queue.get()
    for focus in range(core[0], core[1]): # total indicies counted here
        focus_obj = history[focus] # and call its values
        grav = [0,0,0]
        x = focus_obj[3] # focus location 
        y = focus_obj[4] # variables
        z = focus_obj[5]
        for target in range(len(history)): # compares to all other indices
            if focus == target: # but not itself
                continue
            target_obj = history[target]
            dm = target_obj[1] # target mass and
            dx = target_obj[3] # location variables
            dy = target_obj[4]
            dz = target_obj[5]
            dist = pythagorean(x, y, z, dx, dy, dz) 
            grav = vector_gravity(dm, x, dx, y, dy, z, dz, dist, grav)
        delta_x = grav[0]*const['time']**2+focus_obj[6]*const['time'] # new velocity
        delta_y = grav[1]*const['time']**2+focus_obj[7]*const['time'] # for export
        delta_z = grav[2]*const['time']**2+focus_obj[8]*const['time']
        coord_i = x + delta_x # new location
        coord_j = y + delta_y # for export
        coord_k = z + delta_z
        set_return = [frame + 1, focus_obj[0], focus_obj[1], focus_obj[2], coord_i, coord_j, coord_k, delta_x, delta_y, delta_z] # queue adds frame
        anim_set[focus] = set_return
    queue.put(anim_set)


def pythagorean(x, y, z, dx, dy, dz): # hypotenuse distance between objs' center
    dist = math.hypot(dx - x, dy - y, dz - z) # working
    return dist


def vector_gravity(m, x, dx, y, dy, z, dz, dist, gravity): # newtonian gravity, while t = 1
    acc_x = (-1)*((const['newton'])*m*(x-dx))/(((dist**2)+(const['e']**2))**(3/2)) 
    acc_y = (-1)*((const['newton'])*m*(y-dy))/(((dist**2)+(const['e']**2))**(3/2)) 
    acc_z = (-1)*((const['newton'])*m*(z-dz))/(((dist**2)+(const['e']**2))**(3/2)) 
    gravity = [acc_x + gravity[0], acc_y + gravity[1], acc_z + gravity[2]]
    return gravity


if __name__ == "__main__":
    ret = {}
    anim_set = {}
    li = []
    init_l = []
    export_data = []
    sample_sum = const['processes_nodes'] * const['samples']
    print("initializing...") # print
    processes_n = []
    queue = mp.Manager().Queue()
    queue.put(ret)
    for i in range(const['processes_nodes']):
        p = mp.Process(target=init_dataset, args=(queue, i,))
        p.start()
        processes_n.append(p)
    for p in processes_n:
        p.join()
    ret = queue.get()
    init_l = list(ret.values())
    for initn in range(sample_sum):
        li.append([1, init_l[initn][0], init_l[initn][3], init_l[initn][4], init_l[initn][5]])
    export_data.extend(li)
    pd_dataframe = pd.DataFrame(init_l, columns=["name", "mass", "radius", "lx", "ly", "lz", "vx", "vy", "vz"])
    pd_dataframe.to_csv(r'N:\init1.csv')
    print("done initializing ... [now rendering frames]") # print
    core = []
    processes_n = []
    for i in range(const['processes_nodes']):
        i_process_sum = const['samples'] * i
        k_process_sum = const['samples'] * (i + 1)
        core.append([i_process_sum, k_process_sum])
    for frame in range(1, const['frame_total'] + 1, const['frame_interval']):
        queue.put(anim_set)
        for i in range(const['processes_nodes']):
            p = mp.Process(target=anim_dataset, args=(queue, init_l, core[i], frame,)) # previous_set, anim_set, 
            p.start()
            processes_n.append(p)
        for p in processes_n:
            p.join()
        anim_set = queue.get()
        anim_list = list(ret.values())
        export_list_temp = [] # building list to extend into animation dataset
        for n in range(sample_sum): # ["frame", "name", "x", "y", "z"]
            export_list_temp.append([anim_set[n][0], anim_set[n][1], anim_set[n][4], anim_set[n][5], anim_set[n][6]])
        export_data.extend(export_list_temp)
        return_list_temp = [] # building list for internal use in function
        for m in range(sample_sum): # ["name", "mass", "radius", "lx", "ly", "lz", "vx", "vy", "vz"]
            return_list_temp.append([anim_set[m][1], anim_set[m][2], anim_set[m][3], anim_set[m][4], anim_set[m][5], anim_set[m][6], anim_set[m][7], anim_set[m][8], anim_set[m][9]])
        init_l = return_list_temp
        print(init_l[3])
        print("done with frame", frame) # print
    pd_dataframe = pd.DataFrame(export_data, columns=["frame", "name", "x", "y", "z"])
    pd_dataframe.to_csv(r'N:\anim1.csv')
    print("done with animation") # print
