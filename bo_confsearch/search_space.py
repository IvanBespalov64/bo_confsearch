from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from default_vals import ConfSearchConfig
from coef_calc import CoefCalculator
from db_connector import LocalConnector

class SearchSpace(ABC):
    def __init__(
        self    
    ) -> None:
        """
            Search Space abstract class
        """
        pass

    @abstractmethod
    def configure_search_space(
        self,
        *args,
        **kwargs
    ) -> List[Tuple[float, float]]:
        """
            Determins the search space
            Returns:
                Search space boundary as list of tuples with
                left and right boundaries
        """
        pass

    @abstractmethod
    def get_xyz_block_from_coords(
        self,
        coordinates : List[float]
    ) -> str:
        """
            Configures required conformer in defined search space
            Args:
                coordinates : list of coordinates
            Returns:
                XYZ coords block of required conformer
                
        """
        pass

    @abstractmethod
    def get_orca_constraints_block(
        self,
        coordinates : List[float]
    ) -> str:
        """
            Returns ORCA constaints block for required conformer
            Args:
                coordinates : list of coordinates
            Returns:
                ORCA constraints block

        """
        pass

    @abstractmethod
    def get_coords_from_xyz_block(
        self,
        xyz_block : str
    ) -> List[float]:
        """
            Parse coordinates from xyz_block
            Args:
                xyz_block: XYZ coords block
            Returns:
                list of coordinates in defined search space
        """
        pass

    @property
    @abstractmethod
    def dim(
        self
    ) -> int:
        """
            Calculates number of dimensions of search space
            Returns:
                number of dimensions
        """
        pass

class DefaultSearchSpace(SearchSpace):
    def __init__(
        self,
        mol : Chem.rdchem.Mol,
        config : ConfSearchConfig
    ) -> None:
        self.mol = mol
        self.config = config
   
        self.dihedral_ids = []
        self.mean_func_coefs = []

    def configure_search_space(
        self        
    ) -> List[Tuple[float]]:
        coef_matrix = CoefCalculator(
            mol=Chem.RemoveHs(self.mol),
            config=self.config, 
            dir_for_inps=f"{self.config.exp_name}_scans/", 
            db_connector=LocalConnector(self.config.dihedral_logs)
        ).coef_matrix()

        for ids, coefs in coef_matrix:
            self.dihedral_ids.append(ids)
            self.mean_func_coefs.append(coefs)

        return [[0 for _ in range(len(self.dihedral_ids))], [2*np.pi for _ in range(len(self.dihedral_ids))]]

    @property
    def dim(
        self        
    ) -> int:
        return len(self.dihedral_ids)

    def get_xyz_block_from_coords(
        self,
        coordinates : List[float]
    ) -> str:

        tmp_mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(self.mol), removeHs=False)

        for atom_idxs, value in zip(self.dihedral_ids, coordinates):
            rdMolTransforms.SetDihedralRad(tmp_mol.GetConformer(), *atom_idxs, value)

        return '\n'.join(Chem.MolToXYZBlock(tmp_mol).split('\n')[2:])

    def get_orca_constraints_block(
        self,
        coordinates : List[float]
    ) -> str:
        res = ""
        for atom_idxs, value in zip(self.dihedral_ids, coordinates):
            res += "{ " + f"D {atom_idxs[0]} {atom_idxs[1]} {atom_idxs[2]} {atom_idxs[3]} {value * 180 / np.pi} C" + " }\n"
        return res

    def get_coords_from_xyz_block(
        self,
        xyz_block : str
    ) -> List[float]:

        def dihedral_angle(
            a : np.ndarray, 
            b : np.ndarray, 
            c : np.ndarray, 
            d : np.ndarray
        ) -> float:
            """
                Calculates dihedral angel between 4 points
                Calc of signed dihedral angel in terms of rdkit
                Vars named like in rdkit source code
            """
            
            lengthSq = lambda u : np.sum(u ** 2)

            nIJK = np.cross(b - a, c - b)
            nJKL = np.cross(c - b, d - c)
            m = np.cross(nIJK, c - b)

            res =  -np.arctan2(np.dot(m, nJKL) / np.sqrt(lengthSq(m) * lengthSq(nJKL)),\
                       np.dot(nIJK, nJKL) / np.sqrt(lengthSq(nIJK) * lengthSq(nJKL)))
            return (res + 2 * np.pi) % (2 * np.pi)
       
        coords = np.asarray([list(map(float, line.strip().split()[1:])) for line in xyz_block.split('\n')])

        res = []
        for a, b, c, d in self.dihedral_ids:
            res.append(
                dihedral_angle(
                    coords[a, :],
                    coords[b, :],
                    coords[c, :],
                    coords[d, :]
                )        
            )

        return res 
