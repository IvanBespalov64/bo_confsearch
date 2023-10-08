from numpy import array

res = {0: (array([-0.35211771]), 0), 
       1: (array([3.80133093]), 13), 
       2: (array([3.28684898]), 8), 
       3: (array([-0.00894725]), 30), 
       4: (array([5.27468301]), 35), 
       5: (array([-0.09915781]), 38), 
       6: (array([-1.2582563]), 40)}

raw_res = [res[cur_id] for cur_id in res]
raw_res = sorted(raw_res, key=lambda x : x[0][0])

with open("conformers.xyz", "w") as file:
    for cur in raw_res:
        cur_coords = []
        with open("structs/" + str(cur[1]) + ".xyz", "r") as cur_xyz:
            cur_coords = [line for line in cur_xyz]
        cur_coords[1] = f"Relative E={cur[0][0]} kcal/mol\n"
        file.write("".join(cur_coords) + "\n")

