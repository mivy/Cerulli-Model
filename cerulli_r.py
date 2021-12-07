# Project Cerulli by Mivy

import pandas as pd
import math
import numpy as np
import multiprocessing as mp


const = {
    'newton' : float(6.67408e-11),      # 6.67408e-11  
    'e' : .11,                          # softening function constant  ---     NOTE FOR DEBUGGING USE ONLY !!
    'c' : 299_792_458,                  # speed of light (m/s)
    'frame_total' : 180,                # total number of frames
    'time' : 1e35,                      # universal "time"       ---    ---     NOTE NOT A DIRECT SCALAR TOOL !!
    'p_nodes' : 2,                      # number of processes spawned
    'samples' : 50,                     # number of samples per process
    'mass_p_law' : 10,                # scale the exponent of the mass distribution inverse square [10 default]
    'mass_scalar' : 1e39,                # scale mass of all particles [20:1 dE default] 3 # mass scale at FINAL
    'radi_scalar' : 1e-18,               # scale particle size [visual]
    'distance_scalar' : 1e12,            # scale grid - average distance between stars 149.6 e9 * 860 [1.3e16 default] 
    'velocity_scalar' : 1e18,          # scale frame 0 velocity of particles [positive velocity is -expansion-]
    }


def function_init(queue, core): 
    ret = queue.get()
    for mia in range(const['samples']): # total indicies counted here

        # index
        index = mia + core[0]

        # name
        name = f'n{index}'

        # mass distribution
        p = abs(np.random.default_rng().standard_normal())
        k = .6 + (p*.35) ** const['mass_p_law'] # truncated power law

        # mass
        mass = const['mass_scalar'] * k

        # var
        x = np.random.default_rng().standard_normal()
        y = np.random.default_rng().standard_normal()
        z = np.random.default_rng().standard_normal()

        # location
        cx = x * const['distance_scalar']
        cy = y * const['distance_scalar']
        cz = z * const['distance_scalar']
        
        # velocity
        dx = -1 * const['velocity_scalar'] * 1 / y ** 2
        dy = -1 * const['velocity_scalar'] * 1 / z ** 2
        dz = -1 * const['velocity_scalar'] * 1 / x ** 2

        # shader
        shader = abs(np.random.default_rng().standard_normal()) / 3
        shader = function_particle_shader(mass, shader)
        
        # rotation
        rx = y
        ry = z
        rz = x

        # radius (bpy)
        radius = function_particle_radius(mass, shader)

        # return thread results (in an ordered list)
        ret[index] = [name, mass, radius, shader, cx, cy, cz, dx, dy, dz, rx, ry, rz] 

    # return process results
    queue.put(ret)      


def function_ii(queue_l, rset, i, j, nullif): # [name, mass, radius, shader, cx, cy, cz, dx, dy, dz, rx, ry, rz] 

    time = const['time']
    r = []

    for n in range(i, j):

        dx, dy, dz = 0, 0, 0
        ix, iy, iz = 0, 0, 0
        mx, my, mz = 0, 0, 0
        nx, ny, nz = 0, 0, 0

        bll = len(rset)

        id_refer = rset[n][0]
        mass_refer = rset[n][1]

        flag = False
        for x in range(len(nullif)):
            if id_refer == nullif[x]:
                flag=True
                break

        if (flag == True) or (mass_refer == 0):
            
            # skip impacted object
            queue_l.append(rset[n])

            continue

        shader_refer = rset[n][3]
        radius_refer = rset[n][3]
        cx1 = rset[n][4]
        cy1 = rset[n][5]
        cz1 = rset[n][6]
        dx1 = rset[n][7]
        dy1 = rset[n][8]
        dz1 = rset[n][9]
        rx1 = rset[n][10]
        ry1 = rset[n][11]
        rz1 = rset[n][12]

        mass = mass_refer

        for m in range(bll):

            gx, gy, gz = 0, 0, 0
            cx, cy, cz = 0, 0, 0
            rx, ry, rz = 0, 0, 0
            kx, ky, kz = 0, 0, 0

            id_target = rset[m][0]
            mass_target = rset[m][1]

            # both cant ever be true
            if m == n: 

                continue

            flag = False
            for x in range(len(nullif)):
                if id_target == nullif[x]:
                    flag=True
                    break

            if(flag==True):
                
                continue

            shader_target = rset[m][3]
            radius_target = rset[m][3]
            cx2 = rset[m][4]
            cy2 = rset[m][5]
            cz2 = rset[m][6]
            dx2 = rset[m][7]
            dy2 = rset[m][8]
            dz2 = rset[m][9]
            rx2 = rset[m][10]
            ry2 = rset[m][11]
            rz2 = rset[m][12]

            cx = (cx2 - cx1)
            cy = (cy2 - cy1)
            cz = (cz2 - cz1)

            distance = function_hyp_lookup(r, id_refer, id_target)
            
            if distance == -1:

                distance = function_pythag(cx, cy, cz)
                foo = [[id_target, id_refer, distance]] # NOTE to flip [target] and [reference]
                r.extend(foo)

            coldetect = ( radius_refer + radius_target ) * 2e10

            if distance <= coldetect:

                foo = mass_target / mass_refer
                
                # relative velocity by ratio mass
                ix += ( dx2 + dx1 ) / 2 * foo
                iy += ( dy2 + dy1 ) / 2 * foo
                iz += ( dz2 + dz1 ) / 2 * foo

                mx += ( rx2 + rx1 ) / 2 * foo
                my += ( ry2 + ry1 ) / 2 * foo
                mz += ( rz2 + rz1 ) / 2 * foo

                mass = mass_refer + mass_target

                print(id_target, "impacted with", id_refer)

                nullif.append(id_target)

                # != reversing singularity
                if shader_target > 1:

                    shader = 2

            else:

                # Newtonian gravity
                gx1, gy1, gz1 = function_newton_n(cx, cy, cz, mass_target, distance)     

                try:
                    
                    # calculate gravitational time dialation
                    relative_time = function_ΛGTD(mass_target, distance)

                    # simplify to standard time
                    gx = gx1 * relative_time / time
                    gy = gy1 * relative_time / time
                    gz = gz1 * relative_time / time

                    kx = rx2 * relative_time / time
                    ky = ry2 * relative_time / time
                    kz = rz2 * relative_time / time

                except:

                    # smooth transition
                    gx = gx1
                    gy = gy1
                    gz = gz1

                    kx = rx2
                    ky = ry2
                    kz = rz2
                
            dx += gx
            dy += gy
            dz += gz

            rx += kx
            ry += ky
            rz += kz

        dx = (dx + ix) * time
        dy = (dy + iy) * time
        dz = (dz + iz) * time

        rx = (rx + mx) * time / bll
        ry = (ry + my) * time / bll
        rz = (rz + mz) * time / bll

        nx = dx + cx1
        ny = dy + cy1
        nz = dz + cz1

        # radius
        if mass != mass_refer:

            shader = function_particle_shader(mass, shader_refer)
            radius = function_particle_radius(mass, shader)
            
        else:

            shader = shader_refer
            radius = radius_refer

        tana = [id_refer, mass, radius, shader, nx, ny, nz, dx, dy, dz, rx, ry, rz]

        if n == 4:
            print(tana[4], tana[5], tana[6])

        queue_l.append(tana)


def function_hyp_lookup(dl, n, nt):
    
    d = -1

    # in the list dl
    for i in range(len(dl)):

        # if distance is listed
        if dl[i][0] == n and dl[i][1] == nt:

            # return distance
            d = dl[i][2]

    return d


def function_pythag(dx, dy, dz):

    d = math.hypot(dx, dy, dz)

    return d


def function_particle_shader(m, s):

    try:

        # 0 < s < 1 : temperature
        if 0 <= s <= 1:

            j = m / const['mass_scalar']
            k = ( ( j - .6 ) ** ( 1. / const['mass_p_law'] ) ) / .35
            
            k /= 3.4
            #print(k)

        # black hole
        elif 1 < s <= 2: 

            k = 2

        return k

    except:

        print(s, "err : shader")
    

def function_particle_radius(m, s):

    try:

        if 1 < s <= 2:

            k = ( ( 2 * const['newton'] * m ) / const['c'] ) * 5e-16
            # NOTE [future] scale rotation
            
        elif 0 <= s <= 1:

            k = s

        return k

    except:

        print(s, "err : radius")


def function_newton_n(x, y, z, m, d): # NOTE already additive reciprocal

    dx = const['newton'] * m * x / ( abs( d ) * d ) ** 2
    dy = const['newton'] * m * y / ( abs( d ) * d ) ** 2
    dz = const['newton'] * m * z / ( abs( d ) * d ) ** 2

    return dx, dy, dz


def function_newton_e(x, y, z, m, d):

    dx = 1 * const['newton'] * m * x / ( d ** 2 + const['e'] ** 2 ) ** 3/2
    dy = 1 * const['newton'] * m * y / ( d ** 2 + const['e'] ** 2 ) ** 3/2
    dz = 1 * const['newton'] * m * z / ( d ** 2 + const['e'] ** 2 ) ** 3/2

    return dx, dy, dz


def function_ΛGTD(mass, dist):

    c = 2 * const['newton'] * mass / dist * const['c'] ** 2
    t = const['time'] * math.sqrt(1 - c)

    return t


def function_ΛCDM():
    
    p = 0
    
    return p


if __name__ == "__main__":
    queue = mp.Manager().Queue()
    ret = {}
    result = []

    # for multiprocessing, divide samples into processes
    core = [] 
    for i in range(const['p_nodes']):
        i_process_sum = const['samples'] * i
        k_process_sum = const['samples'] * (i + 1)
        core.append([i_process_sum, k_process_sum])

        # print(27 // 4) # divisor int

    # begin setup
    processes_i = []
    queue.put(ret)
    for i in range(const['p_nodes']):
        pi = mp.Process(target=function_init, args=(queue, core[i],))
        pi.start()
        processes_i.append(pi)

    # resolve processes
    for pi in processes_i:
        pi.join()
    ret = queue.get()

    core = None
    queue = None

    # format data
    rset = list(ret.values())

    # start simulation
    for frame in range(1, const['frame_total'] + 1):
        
        print(f"Calculating F{frame}...")
        
        # samples calc
        quotient, remainder = divmod( len(rset) , const['samples'] )

        # step simulation
        processes_k = []
        queue_l = mp.Manager().list()
        nullif = mp.Manager().list()
        for i in range(quotient):
            
            # samples
            tempvar_a = i * const['samples']
            tempvar_b = ((i+1) * const['samples']) - 1

            pk = mp.Process(target=function_ii, args=(queue_l, rset, tempvar_a, tempvar_b, nullif,))
            pk.start()
            processes_k.append(pk)

        # samples
        tempvar_a = quotient * const['samples']
        tempvar_b = quotient * const['samples'] + remainder

        pk = mp.Process(target=function_ii, args=(queue_l, rset, tempvar_a, tempvar_b, nullif,))
        pk.start()
        processes_k.append(pk)

        # resolve results
        for pk in processes_k:
            pk.join()

        # format data
        queue_l = list(queue_l)

        # collisions
        nullif = list(nullif)
        
        
        flag = False

        for x in range(len(queue_l)):

            rep = len(nullif)

            if flag == True:
                rep = rep - 1

            flag = False

            for y in range(rep):

                if queue_l[x][0] == nullif[y]:

                    # NOTE -1 shader = bpy delete
                    queue_l[x] = [queue_l[x][0], 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    flag = True

                    print(queue_l[x][0], "deleted")

                    break
        
        rset = queue_l

        queue_l = map(lambda x: x[:1]+x[2:], queue_l)
        queue_l = map(lambda x: x[:6]+x[9:], queue_l)
        queue_l = list(queue_l)

        queue_l.extend([['flag', 0, 0, 0, 0, 0]]) # bpy flag # [id_refer, mass, radius, shader, nx, ny, nz, dx, dy, dz, rx, ry, rz]
        
        result.extend(queue_l)
        
    pd_dataframe = pd.DataFrame(result, columns=["id", "r", "c", "x", "y", "z"])
    pd_dataframe.to_csv(r'N:\cerulli-ldata.csv')

    print("complete")
