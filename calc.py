import os
import os.path
import time
import math
from typing import Union
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

from sklearn.cluster import KMeans

import default_vals

from typing import Dict

HARTRI_TO_KCAL = 627.509474063 

USE_ORCA = default_vals.USE_ORCA
ORCA_EXEC_COMMAND = default_vals.ORCA_EXEC_COMMAND
GAUSSIAN_EXEC_COMMAND = default_vals.GAUSSIAN_EXEC_COMMAND
NUM_OF_PROCS = default_vals.DEFAULT_NUM_OF_PROCS
DEFAULT_METHOD = default_vals.DEFAULT_METHOD
ORCA_METHOD = default_vals.DEFAULT_ORCA_METHOD
CHARGE = default_vals.DEFAULT_CHARGE
MULTIPL = default_vals.DEFAULT_MULTIPL

CURRENT_STRUCTURE_ID = 0 # global id for every structure that we would save

WRONG_GEOMETRY = False
SKIP_TRJ = False

#Alias for type of node about dihedral angle 
#that consists of list with four atoms and value of degree
dihedral = tuple[list[int], float]

def load_params_from_config(
    config : Dict[str, object]
) -> None:
    
    global USE_ORCA, ORCA_EXEC_COMMAND, NUM_OF_PROCS, ORCA_METHOD, ORCA, CHARGE, MULTIPL
    
    print(f"Loading calculation config!")
    update_number = 0
    if "spin_multiplicity" in config:
        if not isinstance(config["spin_multiplicity"], int):
            print(f"Spin Multiplicity should be integer! Continue with default value {MULTIPL}")
        else:
            MULTIPL = config["spin_multiplicity"]
            update_number += 1

    if "charge" in config:
        if not isinstance(config["charge"], int):
            print(f"Charge should be integer! Continue with default value {CHARGE}")
        else:
            CHARGE = config["charge"]
            update_number += 1

    if "use_orca" in config:
        if not isinstance(config["use_orca"], bool):
            print(f"use_orca should be bool! Continue with default value {USE_ORCA}")
        elif not config["use_orca"]:
            print("Now only orca calculations supported! Continue with use_orca=True")
        else:
            USE_ORCA = config["use_orca"]
            update_number += 1
    
    if "orca_exec_command" in config:
        if not isinstance(config["orca_exec_command"], str):
            print(f"Orca execution command should be str! Continue with default value {ORCA_EXEC_COMMAND}")
        else:
            ORCA_EXEC_COMMAND = config["orca_exec_command"]
            update_number += 1    

    if "num_of_procs" in config:
        if not isinstance(config["num_of_procs"], int):
            print(f"Number of procs should be integer! Continue with default value {NUM_OF_PROCS}")
        else:
            NUM_OF_PROCS = config["num_of_procs"]
            update_number += 1

    if "orca_method" in config:
        if not isinstance(config["orca_method"], str):
            print(f"Orca method should be str! Continue with default value {ORCA_METHOD}")
        else:
            ORCA_METHOD = config["orca_method"]
            update_number += 1

    print(f"Calculation config loaded! {update_number} params were updated!")

def dist_between_atoms(mol : Chem.rdchem.Mol, i : int, j : int) -> float:
    pos_i = mol.GetConformer().GetAtomPosition(i)
    pos_j = mol.GetConformer().GetAtomPosition(j)
    
    return np.sqrt((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2 + (pos_i.z - pos_j.z) ** 2)

def change_dihedrals(mol_file_name : str,
                     dihedrals : list[dihedral], full_block = False) -> Union[str, None]:
    """
        changinga all dihedrals in 'dihedrals' 
        (degree in rads, numeration starts with zero),
        returns coords block of xyz file
        if no file exists, return None
        if fullBlock is True returns full xyz block
    """
    try:
        mol = Chem.MolFromMolFile(mol_file_name, removeHs = False)
        
        for note in dihedrals:
            atoms, degree = note
            rdMolTransforms.SetDihedralRad(mol.GetConformer(), *atoms, degree)
        
        if(full_block):
            return Chem.MolToXYZBlock(mol)
        return '\n'.join(Chem.MolToXYZBlock(mol).split('\n')[2:])
    except OSError:
        print("No such file!")
        return None

def random_displacement(xyz : str, alpha : float):
    """
        xyz - xyz coord block 
        displace all atoms on the distance 0 <= dr <= alpha
        returns modified xyz coord block
    """
    res = ""
    for line in xyz.split('\n'):
        if line == "":
            break
        atom, coords = line.split()[0], np.array(list(map(float, line.split()[1:])))
        res += atom + " " + " ".join(map(str, coords + alpha * np.random.random(3))) + "\n"
    return res

def to_degrees(dihedrals : list[dihedral]) -> list[dihedral]:
    """
        Convert rads to degrees in dihedrals
    """
    res = []
    for cur in dihedrals:
        a, d = cur
        res.append((a, d * 180 / math.pi))
    print(res)
    return res

def read_xyz(name : str) -> list[str]:
    """
        read coords from 'filename' and return that as a list of strings
    """
    xyz = []
    with open(name, 'r') as file:
        for line in file:
            xyz.append(line)
    return '\n'.join(xyz)

def generate_gjf(coords : str,
                 gjf_name : str,
                 num_of_procs : int,
                 method_of_calc : str,
                 charge : int,
                 multipl : int):
    """
        coords - xyz coords bloack with '\n' sep
        generate .gjf file for calculation energy of 'coords' by 'method_of_calc'
    """
    with open(gjf_name, 'w+') as file:
        file.write("%nprocs=" + str(num_of_procs) + "\n" + "\n")
        file.write("#P " + method_of_calc + "\n" + "\n")
        file.write(gjf_name + "\n" + "\n")
        file.write(str(charge) + " " + str(multipl) + "\n")
        file.write(coords)
        file.write("\n" + "\n")

def generate_default_gjf(coords : str, gjf_name : str):
    """
        generate .gjf with default properties
    """
    generate_gjf(coords, gjf_name, NUM_OF_PROCS, DEFAULT_METHOD,\
                                         CHARGE, MULTIPL)

def generate_oinp(
        coords : str, 
        dihedrals : list[dihedral],
        gjf_name : str, 
        num_of_procs : int, 
        method_of_calc : str,
        charge : int,
        multipl : int,
        constrained_opt : bool = False
    ) -> None:
    """
        generates orca .inp file
    """
    with open(gjf_name, 'w+') as file:
        opt_cmd = "opt"
        if default_vals.ts:
            opt_cmd = "OptTS"
        file.write("!" + method_of_calc + f" {opt_cmd}\n")
        file.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")
        if constrained_opt:
            file.write("%geom Constraints\n")
            dihedrals = to_degrees(dihedrals)
            for cur in dihedrals:
                a, d = cur
                file.write("{ D " + " ".join(map(str, a)) + " " + str(d) + " C }\n")
            file.write("end\n")
            file.write("end\n")    
        file.write("* xyz " + str(charge) + " " + str(multipl) + "\n")
        file.write(coords)
        file.write("END\n")

def generate_default_oinp(coords : str, dihedrals : list[dihedral], oinp_name : str, constrained_opt : bool = False):
    generate_oinp(coords, dihedrals, oinp_name, NUM_OF_PROCS, ORCA_METHOD,\
                                                CHARGE, MULTIPL, constrained_opt=constrained_opt)
def start_calc(gjf_name : str):
    """
        Running calculation
    """	
    if not USE_ORCA:
        os.system(GAUSSIAN_EXEC_COMMAND + " " + gjf_name)
    else:
        sbatch_name = gjf_name.split('/')[-1][:-4] + ".sh"
        os.system("cp sbatch_temp " + sbatch_name)
        os.system("echo \"" + ORCA_EXEC_COMMAND + " " + gjf_name + " > " + gjf_name[:-4] + ".out\"" + " >> " + sbatch_name) 
        os.system("sbatch " + sbatch_name)    

def gjf_to_log_name(gjf_name : str) -> str:
    """
        generating name of log file from gjf file name
    """
    return gjf_name[:-4] + ".log"

def mol_to_gjf_name(mol_file_name : str) -> str:
    """
        generating name of gjf file from mol file name
    """
    return mol_file_name[:-4] + ".gjf"

def mol_to_inp_name(mol_file_name : str) -> str:
    """
        generating name of inp file from mol file name
    """
    return mol_file_name[:-4] + ".inp"

def inp_to_out_name(inp_file_name : str) -> str:
    """
        generating name of out file from inp file name
    """
    return inp_file_name[:-4] + ".out"

def wait_for_the_end_of_calc(log_name : str, timeout):
    """
        waiting fot the end of calculation in gaussian 
        by checking log file every 'timeout' ms
    """
    while True:
        try: 
            with open(log_name, 'r') as file:
                log_file = [line for line in file]               
                if(not USE_ORCA and ("Normal termination" in log_file[-1] or\
                   "File lengths" in log_file[-1])):
                    break
                if(USE_ORCA and ("ORCA TERMINATED NORMALLY" in log_file[-2] or\
                                 "ORCA finished by error" in log_file[-5])):
                    break
        except FileNotFoundError:
            pass
        except IndexError:
            pass
        finally:
            time.sleep(timeout / 1000)

def find_energy_in_log(log_name : str) -> float:
    """
        finds energy of structure in log file
    """
    with open(log_name, 'r') as file:
        if not USE_ORCA:
            en_line = [line for line in file if "SCF Done" in line][0]
            en = float(en_line.split()[4])
        else:
            en_line = [line for line in file if "FINAL SINGLE POINT ENERGY" in line][-1]
            en = float(en_line.split()[4])
        return en
    
def calc_energy(
        mol_file_name : str,
        dihedrals : list[dihedral] = [],
        norm_energy : float = 0, 
        save_structs : bool = True,
        #RANDOM_DISPLACEMENT : bool = False,
        constrained_opt : bool = False,
        force_xyz_block : Union[None, str] = None
    ) -> float:
    """
        calculates energy of molecule from 'mol_file_name'
        with current properties and returns it as float
        Also displace atoms on random distances is RANDOM_DISPLACEMENT = True
    """

    global WRONG_GEOMETRY
    global SKIP_TRJ

    print(f"Calc with save_struct={save_structs}")

    xyz_upd = None
    
    if force_xyz_block:
        xyz_upd = force_xyz_block
    else:
        xyz_upd = change_dihedrals(mol_file_name, dihedrals)

    print(dihedrals)
    print(list(zip(*dihedrals)))

    if(xyz_upd is None):
        WRONG_GEOMETRY = False
        SKIP_TRJ = True
        return None

    if WRONG_GEOMETRY:    
        return 100

    if not USE_ORCA:
        gjf_name = mol_to_gjf_name(mol_file_name)
        log_name = gjf_to_log_name(gjf_name)
        if os.path.isfile(log_name):
            os.system("rm -r " + log_name)     
        generate_default_gjf(xyz_upd, gjf_name)
        start_calc(gjf_name)
        wait_for_the_end_of_calc(log_name, 250)
        res = find_energy_in_log(log_name)
    else:
        inp_name = mol_to_inp_name(mol_file_name)
        out_name = inp_to_out_name(inp_name)
        if os.path.isfile(out_name):
            os.system("rm -r " + out_name)
        generate_default_oinp(xyz_upd, dihedrals, inp_name, constrained_opt=constrained_opt)
        start_calc(inp_name)
        wait_for_the_end_of_calc(out_name, 1000)
        res = find_energy_in_log(out_name) * HARTRI_TO_KCAL - norm_energy
    return res

def load_last_optimized_structure_xyz_block(mol_file_name : str) -> str:
    full_xyz = []
    with open(mol_file_name[:-4] + '.xyz', 'r') as xyz_file:
        for line in xyz_file:
            full_xyz.append(line)
    return ''.join(full_xyz[2:])

def dihedral_angle(a : list[float], b : list[float], c : list[float], d : list[float]) -> float:
    """
       Calculates dihedral angel between 4 points
    """
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    #Next will be calc of signed dihedral angel in terms of rdkit
    #Vars named like in rdkit source code

    lengthSq = lambda u : np.sum(u ** 2)
    
    nIJK = np.cross(b - a, c - b)
    nJKL = np.cross(c - b, d - c)
    m = np.cross(nIJK, c - b)

    res =  -np.arctan2(np.dot(m, nJKL) / np.sqrt(lengthSq(m) * lengthSq(nJKL)),\
                       np.dot(nIJK, nJKL) / np.sqrt(lengthSq(nIJK) * lengthSq(nJKL)))
    return (res + 2 * np.pi) % (2 * np.pi)
       
def parse_points_from_trj(
    trj_file_name : str,
    dihedrals : list,
    norm_en : float, 
    save_structs : bool = True,
    structures_path : str = "structs/", 
    return_minima : bool = True
) -> Union[list[tuple[list[dihedral], float]], tuple[list[tuple[list[dihedral], float]], tuple[list[dihedral], float]]]:
    """
        Parse more points from trj orca file
        returns list of description of dihedrals
        for every point
    """

    global SKIP_TRJ

    if SKIP_TRJ:
        SKIP_TRJ = False
        return []

    print(f"Parsing starts with norm_en={norm_en}, save_struct={save_structs}")

    result = []

    structures = []

    global CURRENT_STRUCTURE_ID

    with open(trj_file_name, "r") as file:
        lines = [line[:-1] for line in file]
        n = int(lines[0])
        for i in range(len(lines) // (n + 2)):
            structures.append("\n".join(lines[i * (n + 2) : (i + 1) * (n + 2)]))
            
            energy = float(lines[i * (n + 2) + 1].split()[-1]) * HARTRI_TO_KCAL - norm_en
            cur_d = []
            for a, b, c, d in dihedrals:
                a_coord = list(map(float, lines[i * (n + 2) + 2 + a].split()[1:]))
                b_coord = list(map(float, lines[i * (n + 2) + 2 + b].split()[1:]))
                c_coord = list(map(float, lines[i * (n + 2) + 2 + c].split()[1:]))
                d_coord = list(map(float, lines[i * (n + 2) + 2 + d].split()[1:]))    
                cur_d.append(dihedral_angle(a_coord, b_coord, c_coord, d_coord))
            result.append((cur_d, energy))
    
    points, obs = list(zip(*result[1:]))

    print(f"Points in trj: {len(result)}")
    #print(result)
   
    num_of_clusters = min(4, len(points))
    print(f"Num of clusters: {num_of_clusters}")

    vals = {cluster_id : (1e9, -1) for cluster_id in range(num_of_clusters)}

    #print(points)
    #print(len(points))

    model = KMeans(n_clusters=num_of_clusters)
    model.fit(points)
    
    for i in range(len(points)):
        cluster = model.predict([points[i]])[0]
        #print(cluster)
        if vals[cluster][0] > obs[i]:
            vals[cluster] = obs[i], i
    #print(len(vals))
    #print(vals)
    print(f"PARSING POINTS, CLUSTER NUM = {num_of_clusters}")
    if save_structs:

        print("SAVING STRUCTS")
        print(f"Saving first struct from trj. Current structure number: {CURRENT_STRUCTURE_ID}")
        with open(structures_path + str(CURRENT_STRUCTURE_ID) + ".xyz", "w") as file:
            file.write(structures[0])
        print(f"saved")
        CURRENT_STRUCTURE_ID += 1

        for cluster_id in vals:
            print(f"saving struct number {CURRENT_STRUCTURE_ID}")
            with open(structures_path + str(CURRENT_STRUCTURE_ID) + ".xyz", "w") as file:
                file.write(structures[vals[cluster_id][1] + 1]) # because points parsed from result[1:]
            print("saved")
            CURRENT_STRUCTURE_ID += 1
    
    return [result[0]] + [(points[vals[cluster_id][1]], vals[cluster_id][0]) for cluster_id in vals], result[-1]
