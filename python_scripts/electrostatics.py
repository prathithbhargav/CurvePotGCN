import glob
import os
import subprocess
import pandas
import warnings
import sys

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


# TODO: Remove hard coded application paths and names
def run_msms(pdb_file, output_directory=None, msms=None):
    """ Calculate triangulated surface using MSMS program """
    basename, _ = os.path.splitext(os.path.basename(pdb_file))

    if output_directory is None:
        output_directory = './'
    elif output_directory[-1] != '/':
        output_directory = output_directory + '/'
    else:
        pass

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    msms = os.path.abspath(msms)
    # TODO: Write a function to convert PDB to xyzrn
    msms_path, msms_executable = os.path.split(msms)

    with open(output_directory + basename + '.xyz', 'w') as xyz_file:
        subprocess.run([msms_path + '/pdb_to_xyzrn', os.path.abspath(pdb_file)],
                       stdout=xyz_file, cwd=msms_path)

    # TODO: MSMS is not free for commercial use, replace in the future
    # and also address the discrepancy between surface from MSMS and the
    # surface used by Delphi
    subprocess.run([msms, "-if", basename + ".xyz", "-of", basename,
                    "-probe_radius", "1.5","-de","3.0"], cwd=output_directory)
    return output_directory + basename + '.vert'


def run_delphi(pdb_file, output_directory, output_filename,
    delphi_path, radius_file=None, charge_file=None, pqr=False, grid_size=101, surface=None, center=False):
    """ Run Delphi on protein surface created by MSMS program """
    # TODO: Rewrite using template string
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    parameters = [
        f'scale=2.0',
        f'perfil=70.0',
        f'salt=0.15',
        f'exdi=80',
        f'linit=2000',
        f'maxc=0.0000000001',
    #    f'out(phi, file="{output_directory}/{output_filename}.cube", format="cube")',
    ]

    if pqr:
        parameters += [f'in(pdb,file="{pdb_file}")']
    else:
        this_script_path = os.path.dirname(os.path.realpath(__file__))
        if radius_file is None:
            radius_file = this_script_path + '/amber.siz'

        if charge_file is None:
            charge_file = this_script_path + '/amber.crg'

        parameters += [
            f'in(pdb,file="{pdb_file}")',
            f'in(siz,file="{radius_file}")',
            f'in(crg,file="{charge_file}")'
        ]

    if center:
        parameters += ['acenter(0,0,0)']
    if surface:
        parameters += [
            f'in(frc,file="{surface}")',
            f'out(frc, file="{output_directory}/{output_filename}.pot")',
            f'site(Atom, Potential, Reaction, Coulomb, Field)',
        ]
    print('\n'.join(parameters) + '\n', file=open(f'{output_filename}_tmp.prm', 'w'))
    subprocess.run([delphi_path, f'{output_filename}_tmp.prm'])
    os.remove(f'{output_filename}_tmp.prm')


def parse_electrostatic_potential(potential_file):
    df = pandas.read_fwf(potential_file, skiprows=12, skipfooter=2,
                         dtype={'resSeq': int}, engine='python',
                         names=['name', 'resName', 'chainID', 'resSeq',
                                'potential', 'reaction', 'coulomb',
                                'Ex', 'Ey', 'Ez'
                         ],
                         widths=[5, 3, 3, 9, 10, 10, 10, 10, 10, 10]
    )

    if df['chainID'].isnull().values.any():
        warnings.warn('Chain ID missing. '
                      'Results may be incorrect for multichain proteins. '
                      'Check if the structure had proper chain IDs')

    df['chainID'].fillna('A', inplace=True)
    output = {}
    chain = 0
    for _, data in df.groupby(['chainID'], as_index=False):
        grouped_df = data.groupby(['resSeq'], as_index=False)['potential']
        potential = grouped_df.sum()
        potential.rename(columns={'potential':'total'}, inplace=True)
        potential['mean'] = grouped_df.mean()['potential']
        output[f'chain {chain}'] = potential
        chain += 1
    return output
from Bio import PDB

def get_protein_sequence_length(pdb_file):
    # Create a PDB parser object
    parser = PDB.PDBParser(QUIET=True)

    # Parse the PDB file
    structure = parser.get_structure("protein", pdb_file)

    # Iterate through the structure and find the length of the protein sequence
    protein_length = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    protein_length += 1

    return protein_length


# Call the function to get the length of the protein sequence



input_folder = sys.argv[1]
output_folder = sys.argv[2]
pdb_files = glob.glob(f'{input_folder}/*.pdb')
#pdb_files=['10gs_cleaned.pdb']
surface = True
output = 'output/'
msms= '/scratch/prathith/software/msms/msms.x86_64Linux2.2.6.1'
delphi = '/scratch/prathith/software/delphicpp_v8.5.0_release'
this_script_path = os.path.dirname(os.path.realpath(__file__))

for pdb_file in pdb_files:
    output_filename = os.path.splitext(os.path.basename(pdb_file))[0]
    sequence_length = get_protein_sequence_length(pdb_file)
    if sequence_length > 1000:
        grid_no = 450
    else:
        grid_no = 350
    surface_file = None
    if surface:
        vert_file = run_msms(pdb_file=pdb_file,
                                output_directory=output,
                                msms=msms)
        surface_file = f"{output}/{output_filename}_surf.pdb"
        # TODO: remove this subprocess run with a function
        subprocess.run(['python', f'{this_script_path}/vert2pdb.py',
                        vert_file, pdb_file, '-o', surface_file])

    run_delphi(pdb_file=pdb_file,
            output_directory=output,
            output_filename=output_filename,
            surface=surface_file,
            delphi_path=delphi,
            grid_size=grid_no)
