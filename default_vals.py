from dataclasses import dataclass

@dataclass
class ConfSearchConfig:
    mol_file_name : str
    spin_multiplicity : int = 1
    charge : int = 0
    use_orca : bool = True
    orca_exec_command : str = "/opt/orca5/orca"
    num_of_procs : int = 8
    orca_method : str = "lda sto-3g"
    ts : bool = False
    
#USE_ORCA=True
#ORCA_EXEC_COMMAND = "/opt/orca5/orca"
#GAUSSIAN_EXEC_COMMAND = "srung"
#DEFAULT_NUM_OF_PROCS = 8
#DEFAULT_METHOD = "RHF/STO-3G"
#DEFAULT_ORCA_METHOD = "r2SCAN-3c TightSCF"
#DEFAULT_ORCA_METHOD = "M062X 6-31+G(d) CPCM(water)"
#DEFAULT_CHARGE = 0
#DEFAULT_MULTIPL = 1
#ts = False 
