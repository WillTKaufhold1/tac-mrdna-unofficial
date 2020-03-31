#!/home/wtk23/anaconda3/bin/python

import mrdna
import numpy as np
import argparse

from mrdna.readers import segmentmodel_from_lists
from mrdna.readers import segmentmodel_from_pdb as read_file
from mrdna.simulate import multiresolution_simulation as simulate
from mrdna import SegmentModel

# accept command line argument of number of ssDNA.


parser = argparse.ArgumentParser(prog="mrdna",
				 description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('-o','--output-prefix', type=str, default=None,
                    help="Name for your job's output")
parser.add_argument('-d','--directory', type=str,  default=None,
                    help='Directory for simulation; does not need to exist yet')
parser.add_argument('-g','--gpu', type=int, default=0,
                    help='GPU for simulation; check nvidia-smi for availability')
parser.add_argument('--debye-length', type=float, default=None,
                    help='Adjust the electrostatic repulsion matching osmotic pressure data from Rau and Parsegian under 25 mM MgCl2 condition with a Debye-Hueckel correction from the default 11.1 Angstroms.')

parser.add_argument('--temperature', type=float, default=295,
                    help='Temperature in Kelvin.')

parser.add_argument('--dimensions', type=str, default=None,
                    help='Comma-separated sides of box in Angstroms.')

parser.add_argument('--sequence-file', type=str, default=None,
                    help='Sequence of longest strand.')

parser.add_argument('--output-period', type=float, default=1e4,
                    help='Simulation steps between DCD frames')
# parser.add_argument('--minimization-steps', type=float, default=0,
#                     help='Simulation steps between DCD frames')
parser.add_argument('--coarse-local-twist', action='store_true',
                    help='Use local twist for coarsest representation?')
parser.add_argument('--fix-linking-number', action='store_true',
                    help='Fix the linking number so that it cannot change once local twist is introduced?')
parser.add_argument('--coarse-steps', type=float, default=1e7,
                    help='Simulation steps for coarse model (200 fs/step)')
parser.add_argument('--fine-steps', type=float, default=1e7,
                    help='Simulation steps for fine model (50 fs/step)')

parser.add_argument('--oxdna-steps', type=float, default=None,
                    help='Perform an oxDNA simulation instead of creating atomic model')
parser.add_argument('--oxdna-output-period', type=float, default=1e4,
                    help='Simulation steps between oxDNA configuration and energy output')
parser.add_argument('--coarse-bond-cutoff', type=float, default = 0,
                    help='Ignore bonds beyond this cutoff during first step of simulation; a value of 0 implies bonds are not ignored')

parser.add_argument('--crossover-to-intrahelical-cutoff', type=float, default=-1,
                    help='Set the distance (in Angstroms) beyond which crossovers between helix ends are converted to intrahelical connections; a negative value means no crossovers will be converted')

parser.add_argument('--backbone-scale', type=float, default=1.0,
                    help='Factor to scale DNA backbone in atomic model; try 0.25 to avoid clashes for atomistic simulations')

parser.add_argument('--run-enrg-md', action='store_true',
                    help='Perform the ENRG-MD simulation?')

parser.add_argument('--debug', action='store_true',
                    help='Run through the python debugger?')

parser.add_argument('--draw-cylinders', action='store_true',
                    help='Whether or not to draw the cylinders')
parser.add_argument('--draw-tubes', action='store_true',
                    help='Whether or not to draw the tubes')

parser.add_argument('--hj-equilibrium-angle', type=float, default=0.0,
                    help='Specify an equilibrium dihedral angle for the Holliday junction angle potential; the default value works well for origami')


args = parser.parse_args()
###############


import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

def read_top():
    with open('./prova.top','r') as f:
        data = list(f)
    split = list(map(lambda x : x.split(), data[1:]))
    return split

def read_dat():
    with open('./prova3322.conf', 'r') as f:
        data = list(f)
    raw_dat = np.array(list(map(lambda x : np.array(x.split()).astype(dtype=float), data[3:])))
    return raw_dat

class Nucleotide():
    def __init__(self,position,normal_vector,bb_vector,identity,strand_id,three_prime,five_prime):
        self.position = position*0.85
        self.normal_vector = normal_vector
        self.bb = bb_vector
        self.identity = identity
        self.strand_id = int(strand_id)
        self.three_prime = int(three_prime)
        self.five_prime = int(five_prime)
        
        self.rot_vector = np.cross(self.bb,self.normal_vector)
        
    def get_three_prime_nucleotide(self):
        return self.three_prime
    
    def get_three_prime_stack(self,nts,vec_combs):
        
        self.expected_stack_pos = (self.position + 
                                   self.normal_vector * vec_combs[0] +
                                   self.bb * vec_combs[1] + 
                                   self.rot_vector * vec_combs[2])
        
        all_locs = np.array([nts[i].position for i in nts])
        
        kd = KDTree(all_locs)
        
        possibilities = kd.query_radius(self.expected_stack_pos.reshape(1,-1), r = 0.3)
        
        if len(possibilities[0]) == 0:
            return -1
        elif len(possibilities[0]) == 1:
            return possibilities[0][0]
        else:
            print ("messed up stacking")
            
    def get_basepair(self,nts):
        
        self.expected_pos = (self.position + 
                                   self.bb * 1)
        
        all_locs = np.array([nts[i].position for i in nts])
        
        kd = KDTree(all_locs)
        
        possibilities = kd.query_radius(self.expected_pos.reshape(1,-1), r = 0.3)
        
        if len (possibilities[0]) == 0:
            return -1
        else:
            return possibilities[0][0]

        
positions = read_dat()[:,:3]
bb_vectors = read_dat()[:,3:6]
normal_vectors = read_dat()[:,6:9]
top = read_top()

nts = {}

for index,(pos,vec,bb_vec,t) in enumerate(zip(positions,normal_vectors,bb_vectors,top)):
    nts[index] = Nucleotide(
                        pos,-vec,bb_vec,t[1],t[0],t[2],t[3]
                                )

from scipy.optimize import minimize

def fun_to_minimize(vec,id1,id2):
    A,B,C = vec
    pred_pos = nts[id1].position + A*nts[id1].normal_vector + B*nts[id1].bb + C*nts[id1].rot_vector
    true_pos = nts[id2].position
    
    return np.sqrt(np.sum((true_pos-pred_pos)**2))

stack_combo = minimize(fun_to_minimize, [1,1,1],args = (0,1)).x
stack_combo = np.array([0.34009418, 0.09418847, 0.31168504])

bp_combo = minimize(fun_to_minimize, [1,1,1], args = (0,1146)).x


mrdna_positions = []
mrdna_bps = []
mrdna_three_prime = []
mrdna_stacking = []

for i in nts:
    mrdna_positions.append(nts[i].position*10)
    mrdna_bps.append(nts[i].get_three_prime_stack(nts,bp_combo))
    mrdna_three_prime.append(nts[i].get_three_prime_nucleotide())
    mrdna_stacking.append(nts[i].get_three_prime_stack(nts,stack_combo))



from mrdna.readers.segmentmodel_from_lists import model_from_basepair_stack_3prime

model = model_from_basepair_stack_3prime(
    mrdna_positions,
    mrdna_bps,
    mrdna_stacking,
    mrdna_three_prime
)

breakpoint()

run_args = dict(
            model = model,
            output_name = "out",
            directory = args.directory,
            minimization_output_period = int(args.output_period),
            coarse_local_twist = args.coarse_local_twist,
            fix_linking_number = args.fix_linking_number,
            bond_cutoff = args.coarse_bond_cutoff,
            coarse_output_period = int(args.output_period),
            fine_output_period = int(args.output_period),
            minimization_steps = 0, # int(args.minimization_steps),
            coarse_steps = int(args.coarse_steps),
            fine_steps = int(args.fine_steps),
            backbone_scale = args.backbone_scale,
            oxdna_steps = args.oxdna_steps,
            oxdna_output_period = args.oxdna_output_period,
            run_enrg_md = args.run_enrg_md
        )

simulate( **run_args )

