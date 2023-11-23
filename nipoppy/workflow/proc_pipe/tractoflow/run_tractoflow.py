import argparse
import os
import shutil
from pathlib import Path
import json
import subprocess
import tempfile
import re
import warnings
from itertools import compress

import numpy as np
# from scipy import stats  ## for mode operation?

import nibabel as nib
from bids import BIDSLayout, BIDSLayoutIndexer

import nipoppy.workflow.logger as my_logger

# Author: bcmcpher
# Date: 15-Jun-2023
fname = __file__
CWD = os.path.dirname(os.path.abspath(fname))

# env vars relative to the container.

MEM_MB = 4000


class BIDSSubjectSession:

    def __init__(self, global_configs, bids_dir, participant_id, session_id, use_bids_filter=False, logger=None):

        self.configs = global_configs
        self.bids_dir = bids_dir
        self.participant_id = participant_id
        self.session_id = session_id

        # load from global configs
        DATASET_ROOT = global_configs["DATASET_ROOT"]

        # load the bids filter if it's called
        if use_bids_filter:

            # path to where pipeline specific filters _should_ be
            bidf_path = Path(f"{DATASET_ROOT}", 'proc', 'bids_filter_tractoflow.json')
            # can change to a path to load an arbitrary filter

            # if a filter exists
            if os.path.exists(bidf_path):
                logger.info(' -- Expected bids_filter.json is found.')
                f = open(bidf_path)
                self.bids_filter = json.load(f)  # load the json as a dictionary
                f.close()
                # validate that the dictionary has valid (or at least expected) fields in it
                # validated dictionary fields can be parsed for more complete ignore flags below

            else:
                logger.info(' -- Expected bids_filter.json is not found.')
                self.bids_filter = {}  # make it empty

        else:
            logger.info(' -- Not using a bids_filter.json')

        # because why parse subject ID the same as bids ID?
        subj = participant_id.replace('sub-', '')
        self.subj = subj

        # build a regex of anything not subject id
        srx = re.compile(f"sub-(?!{subj}.*$)")

        # # from mriqc parsing
        # ignore_paths = [
        #     # Ignore folders at the top if they don't start with /sub-<label>/
        #     re.compile(r"^(?!/sub-[a-zA-Z0-9]+)"),
        #     # Ignore all modality subfolders, except for func/ or anat/
        #     re.compile(
        #         r"^/sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/"
        #         r"(beh|fmap|pet|perf|meg|eeg|ieeg|micr|nirs)"
        #     ),
        #     # Ignore all files, except for the supported modalities
        #     re.compile(r"^.+(?<!(_T1w|_T2w|bold|_dwi))\.(json|nii|nii\.gz)$"),
        # ]

        logger.info('Building Single Subject BIDS Layout...')

        # build a BIDSLayoutIndexer to only pull subject ID
        bidx = BIDSLayoutIndexer(ignore=[srx])

        # parse bids directory with subject filter
        self.layout = BIDSLayout(bids_dir, indexer=bidx)
        # check if DB exists on disk first? BIDSLayout(database_path=var)? where is this saved?
        # should this be made / updated as part of BIDS-ification of dicoms?

    def get_modality(self, suffix):

        if self.bids_filter:
            return self.layout.get(**self.bids_filter[suffix])
        else:
            return self.layout.get(suffix=suffix)

    def get_modality_volumes(self, suffix):

        if self.bids_filter:
            return self.layout.get(**self.bids_filter[suffix], extension=".nii.gz")
        else:
            return self.layout.get(suffix=suffix, extension=".nii.gz")

    def get_dwi_files(self):

        dwi_files = self.get_modality_volumes("dwi")

        dwif = []
        for dwi in dwi_files:
            print("-"*75)
            print(f"Parsing data file: {dwi.filename}")
            dwif.append(BIDSdwi(dwi, self.layout))

        return dwif


def max_lmax(ndirs, symmetric=True):

    # sanity check inputs
    if ndirs < 6:
        warnings.warn("There are not enough directions to model a tensor.")
    elif ndirs <= 0:
        raise ValueError('A positive number of directions must be passed.')

    # get continuous (odd and even) harmonic order
    lmax = int(np.floor((-3 + np.sqrt(1 + 8 * ndirs)) / 2.0))

    # if harmonic order is forced to be even (symmetric)
    if symmetric:

        # force an even harmonic order
        if lmax % 2 == 1:
            return (lmax - 1)

    # otherwise return the even harmonic order
    return (lmax)


def bval_bvec_data(bval, bvec, b0thr=45, stol=49, tshell=1000, symmetric=True):

    # call the function that rounds bvals
    rval = shell_bvals(bval, b0thr=b0thr, stol=stol)

    # get the unique non-zero values
    bunq = np.unique(rval)

    # get the default tensor shell
    dten = default_tensor_shell(bval, bvec, tshell)

    # get the "theoretical" data lmax based on directions regardless of shell(s)
    dlmax = max_lmax(bvec[:, rval > 0].shape[1], symmetric)

    # if the data is multishell
    if len(np.unique(rval)) > 2:

        print(f"Multishell data has shells b = {bunq[1:]}")

        # preallocate lists for by-shell values
        mlmax = []
        mldir = []

        # for every non-zero shell
        for shell in bunq[1:]:

            # pull the shells
            tndir = rval == shell
            # b0idx = rval == 0

            # check that directed vectors are unique
            tvec = bvec[:, tndir]
            tdir = np.unique(tvec, axis=0)  # this needs a better tolerance
            mldir.append(tdir.shape[1])

            # compute and print the maximum lmax per shell
            tlmax = max_lmax(tdir.shape[1], symmetric)
            mlmax.append(tlmax)
            print(f" -- Shell b = {int(shell)} has {tdir.shape[1]} directions capable of a max lmax: {tlmax}")

        # get the highest possible shell
        plmax = max(mlmax)

        print(f"The highest theoretical lmax in the data is: {dlmax}")
        print(f"The maximum lmax for any one shell is: {plmax}")

    # if the data is single shell
    else:

        # pull the values for the single shell
        shell = bunq[1]
        tndir = rval == shell
        tvec = bvec[:, tndir]
        tdir = np.unique(tvec, axis=0)
        plmax = dlmax

        # print an equivalent output
        print(f"Single shell data has shell b = {int(shell)}")
        print(f" -- Shell b = {int(shell)} has {tdir.shape[1]} directions capable of a max lmax: {plmax}")
        print(f"The maximum lmax for the data is: {plmax}")

    return (plmax, dlmax, dten)


def default_tensor_shell(bval, bvec, tshell=1000):

    # set up unique shell valuse and set output shell to None
    ushell = np.unique(bval)
    oshell = None

    # while valid output shell not found
    while oshell is None:

        # safety valve if all shells in input are not sufficient
        if len(ushell) == 0:
            raise ValueError("All candidate shells removed.")

        # find the closest shell to target
        cshell = ushell[np.abs(ushell - tshell) == np.min(np.abs(ushell - tshell))]

        # if the closest shell doesn't have enough directions
        if bvec[:, bval == cshell].shape[1] < 6:

            # remove it from ushell so loop can try again
            ushell = np.delete(ushell, np.where(ushell == cshell))

        # otherwise
        else:

            # set the output
            oshell = cshell

    return oshell


def shell_bvals(bval, b0thr=45, stol=49):

    # this can be too aggressive in rounding shells w/ ##50 values
    # bout = np.round(bval, -2)

    # add a check to determine is QSI data is passed instead

    # sort them
    sval = np.sort(bval)

    # zero values below threshold
    sval[sval <= b0thr] = 0

    # indices into sorted values where change bigger than tolerance occurs
    bidx = np.where(np.diff(sval) > stol)[0] + 1

    # get the values that exist w/in each shell
    shell = np.split(sval, bidx)

    # np array of the corrected / rounded values
    fixed = np.asarray([np.mean(np.round((x) / 10) * 10) for x in shell])

    # create a vector of the "effective" bvals
    bout = fixed[abs(bval[None, :] - fixed[:, None]).argmin(axis=0)]
    # these won't be passed, but used to determine shell labels / features

    # return shelled bvals
    return (bout)


class BIDSdwi():

    def __init__(self, dwi, layout, b0thr=45, stol=50):

        # store the default thresholds
        self.b0thr = b0thr
        self.stol = stol

        # set some default properties
        self.directed = False
        self.dir_ref = False
        self.b0s = False

        # path to the files and full filename
        self.dwi_path = dwi.dirname
        self.dwi_file = dwi.path

        # pull the bval / bvec files
        cbval = layout.get_bval(dwi)
        cbvec = layout.get_bvec(dwi)

        # attempt to set direction data

        # if bval file exists
        if cbval:

            # set the file
            self.bval_file = cbval

            # try and load the values, fail w/ error
            try:
                tbval = np.loadtxt(self.bval_file)
                if tbval.ndim != 1:
                    raise ValueError("bval is not a 1D array.")
                self.bval = tbval
            except ValueError:
                raise ValueError("bval file unable to be read.")

        # otherwise set as empty
        else:
            self.bval_file = None
            self.bval = None

        # if bvec file exists
        if cbvec:

            # set the file
            self.bvec_file = cbvec

            # try and load the file, fail w/ error
            try:
                tbvec = np.loadtxt(self.bvec_file)
                if tbvec.ndim != 2:
                    raise ValueError("bvec file is not a 2D array.")
                    if tbvec.shape[0] != 3:
                        raise ValueError("bvec file does not have the correct number of rows.")
                self.bvec = tbvec
            except ValueError:
                raise ValueError("bvec file unable to be read.")

        # otherwise set as empty
        else:
            self.bvec_file = None
            self.bvec = None

        # check if bval / bvec are both not None and the same dimension
        if (self.bval is not None) & (self.bvec is not None):
            if (self.bval.shape[0] != self.bvec.shape[1]):
                raise ValueError("The dimensions of the bval and bvec file do not match.")
        # should error if only 1 is missing

        # determine if the file is "directed" or not

        # if bval / bvec exists
        if (self.bval is not None) & (self.bvec is not None):

            # if there are any non-zero directions
            if any(self.bval > b0thr):

                # if there is at least a tensor model worth of directed info
                if self.bvec[:, self.bval > b0thr].shape[1] >= 6:

                    # it is directed
                    self.directed = True

                # if there are arbitrary weighted directions
                else:
                    # check if there's exactly 3 - probably reference directions
                    # i.e. shell weighting on (x, y, z) axes w/ b0s
                    if np.unique(self.bvec[:, self.bval > b0thr]).shape[1] == 3:
                        self.dir_ref = True

                    # it's still not "directed", but it's more than empty
                    self.b0s = True

            # bval is all 0
            else:
                self.b0s = True

        # there's no bval / bval files
        else:
            self.b0s = True

        # load the data itself into the object
        # probably a better way to get the sidecar path than this...
        self.json_file = self.dwi_file.replace(".nii.gz", ".json")
        self.json = dwi.get_metadata()

        # inspect the image volume
        try:
            self.dwi = nib.load(dwi)
        except ValueError:
            raise ValueError("DWI file is not a valid nifti file.")

        # pull the voxel dimensions
        self.voxmm = np.abs(np.diag(self.dwi.affine)[:3])

        # check if all voxel dimensions are within a tolerance
        if np.max(np.abs(np.mean(self.voxmm) - self.voxmm)) < 0.10*np.abs(np.mean(self.voxmm)):
            self.isotropic = True
        else:
            self.isotropic = False
        # what tolerance should be used here? how can it be set by user?

        # check image affine
        # https://community.mrtrix.org/t/mrconvert-flips-gradients/581/5
        det = np.linalg.det(self.dwi.affine)
        if det < 0:
            self.orientation = "radiological"
        else:
            warnings.warn("bvec x-axis may need to be flipped.")
            self.orientation = "neurological"

        # store axis code of the file
        self.aff2axcodes = nib.aff2axcodes(self.dwi.affine)

        # check file dimension
        if self.dwi.ndim == 4:
            self.nvol = self.dwi.shape[-1]
        elif self.dwi.ndim == 3:
            self.nvol = 1
        else:
            raise ValueError("The DWI file does not have valid dimension(s): {self.dwi.shape}")

        # basic validity check of the loaded data
        if self.bval.shape[0] != self.nvol:
            raise ValueError("The dimensions of the bval/bvec files do not match the image.")
        else:
            print("All data has matching dimensions.")

        # if it's a directed volume
        if self.directed:

            # parse lmax features
            self.lmax, self.max_lmax, self.tensor_shell = bval_bvec_data(self.bval, self.bvec)

        else:

            # set lmax features to none
            self.lmax = None
            self.max_lmax = None
            self.tensor_shell = None

    @property
    def shells(self):
        return np.unique(self.bval)

    @property
    def nshell(self):
        return len(self.shells)

    @property
    def phaseEncoding(self):
        return self.json["PhaseEncodingDirection"]

    @property
    def shape(self):
        return self.dwi.shape

    @property
    def ndirs(self):
        return self.bval[self.bval > self.b0thr].shape[0]

    @property
    def udirs(self):
        return np.unique(self.bvec[:, self.bval > self.b0thr], axis=1)

    @property
    def readout(self):
        return self.json["TotalReadoutTime"]

    @property
    def nb0s(self):
        return self.bval[self.bval < self.b0thr].shape[0]

    def get_phaseEncoding(self, notation="world", directed=True):
        # i/j/k to x/y/z to LR/AP/SI

        # pull phase encoding from sidecar
        ped = self.phaseEncoding
        # axc = self.aff2axcodes  # is this helpful for setting dimension?

        # if it's undirected, drop any possible sign
        if not directed:
            ped = ped[0]
        # this will return just x/y/z w/o sign when only encoding axis is required (TractoFlow)

        # if it's an ijk coordinate, just return it
        if notation == "voxel":
            out = ped

        # if it's an xyz coordinate, convert to mm notation
        elif notation == "world":
            if "i" == ped:
                out = "x"
            elif "i-" == ped:
                out = "x-"
            elif "j" == ped:
                out = "y"
            elif "j-" == ped:
                out = "y-"
            elif "k" == ped:
                print("This is probably an error.")
                out = "z"
            elif "k-" == ped:
                print("This is probably an error.")
                out = "z-"
            else:
                raise ValueError(f"Invalid phaseEncoding stored: {ped}")

        # if it's an axis, convert to labels
        elif notation == "axis":
            if "i" == ped:
                out = "RL"
            elif "i-" == ped:
                out = "LR"
            elif "j" == ped:
                out = "PA"
            elif "j-" == ped:
                out = "AP"
            elif "k" == ped:
                print("This is probably an error.")
                out = "IS"
            elif "k-" == ped:
                print("This is probably an error.")
                out = "SI"
            else:
                raise ValueError(f"Invalid phaseEncoding stored: {ped}")

        # print a polite error for invalid return type
        else:
            raise ValueError("Invalid formatting requested. Options are: 'voxel', 'world', 'axis'")

        return out

# if any of the files are directed
directed = [x.directed for x in zz2]
if any(directed):
    dfiles = list(compress(zz2, directed))
else:
    dfiles = []
    print("No files with directed diffusion data.")

# if any files are b0s
b0files = [x.b0s for x in zz2]
if any(b0files):
    bfiles = list(compress(zz2, b0files))
else:
    bfiles = []
    print("No files with only b0 data.")

# get the n of directed / b0 files
ndvols = len(dfiles)
b0vols = len(bfiles)


# get the phase axis and direction of directed files
dphaseAxis = [x.phaseEncoding[0] for x in dfiles]
dphaseDirection = [x.phaseEncoding for x in dfiles]

# get the phase axis and direction of the b0 files
bphaseAxis = [x.phaseEncoding[0] for x in bfiles]
bphaseDirection = [x.phaseEncoding for x in bfiles]

# determine the number of directed volumes
if ndvols == 1:
    print("There's one directed sequence. Use that.")
    fpe_dmri = dfiles[0]
    if b0vols > 0:
        print("There's at least 1 b0 file to check.")
        # pick / average together b0s of opposite phase encoding
        # if option, extract b0s to a new file
        rpe_dmri = bfiles[0]
    else:
        rpe_dmri = None
        print("Run the single volume w/o topup.")

elif ndvols == 2:
    print("There's 2 directed sequences. Figure out if TopUp is possible.")
    if dphaseAxis[0] == dphaseAxis[1]:
        if dphaseDirection[0] != dphaseDirection[1]:
            # they're opposite phase encodings - topup will happen
            # if they have the same number of volumes you can do the average thing
            if dfiles[0].ndirs == dfiles[1].ndirs:
                print(" -- They have the same n directions")
                if all(np.isclose(dfiles[0].udirs, dfiles[1].udirs)):
                    print(" -- Sequences are mirrored and can be averaged during topup.")
                    fpe_dmri = dfiles[0]  # order doesn't matter because they're identical
                    rpe_dmri = dfiles[1]  # sequences in opposite phase encodings
            else:
                print(" -- They have different n directions")
                # pick the larger sequence, pull b0s from the other
                fpe_dmri = dfiles[np.argmax([x.ndirs for x in dfiles])]
                rpe_dmri = dfiles[np.argmin([x.ndirs for x in dfiles])]
                # if option, extract b0s to a new file
        else:
            print("Same phase encoding direction")
            # they're the same phase encoding
            # check if it's 2 single shell files that can be merged?
            # - if directions are too close it's probably a test / retest
            # generally this is an acquisition error
    else:
        print("Different encoding axes. Wierd.")
        # both files are different encoding axis
        # process the larger one? fail b/c that's really weird?

elif ndvols > 2:
    # probably update the filter to better target
    print("There's more than 2 directed sequences. Split shells to merge?")

else:
    print("Failed real hard. Negative number of files found?")



def parse_data(global_configs, bids_dir, participant_id, session_id, use_bids_filter=False, logger=None):
    """ Parse and verify the input files to build TractoFlow's simplified input to avoid their custom BIDS filter
    """

    ## load from global configs
    DATASET_ROOT = global_configs["DATASET_ROOT"]

    ## load the bids filter if it's called
    if use_bids_filter:

        ## path to where pipeline specific filters _should_ be
        bidf_path = Path(f"{DATASET_ROOT}", 'proc', 'bids_filter_tractoflow.json') 
        ## can change to a path to load an arbitrary filter

        ## if a filter exists
        if os.path.exists(bidf_path):
            logger.info(' -- Expected bids_filter.json is found.')
            f = open(bidf_path)
            bidf = json.load(f) ## load the json as a dictionary
            f.close()
            ## validate that the dictionary has valid (or at least expected) fields in it
            ## validated dictionary fields can be parsed for more complete ignore flags below
            
        else:
            logger.info(' -- Expected bids_filter.json is not found.')
            bidf = {} ## make it empty
            
    else:
        logger.info(' -- Not using a bids_filter.json')

    ## because why parse subject ID the same as bids ID?
    subj = participant_id.replace('sub-', '')

    ## build a regex of anything not subject id
    srx = re.compile(f"sub-(?!{subj}.*$)")

    ## from mriqc parsing
    # ignore_paths = [
    #     # Ignore folders at the top if they don't start with /sub-<label>/
    #     re.compile(r"^(?!/sub-[a-zA-Z0-9]+)"),
    #     # Ignore all modality subfolders, except for func/ or anat/
    #     re.compile(
    #         r"^/sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/"
    #         r"(beh|fmap|pet|perf|meg|eeg|ieeg|micr|nirs)"
    #     ),
    #     # Ignore all files, except for the supported modalities
    #     re.compile(r"^.+(?<!(_T1w|_T2w|bold|_dwi))\.(json|nii|nii\.gz)$"),
    # ]

    logger.info('Building Single Subject BIDS Layout...')

    ## build a BIDSLayoutIndexer to only pull subject ID
    bidx = BIDSLayoutIndexer(ignore=[srx])

    ## parse bids directory with subject filter
    layout = BIDSLayout(bids_dir, indexer=bidx)
    ## check if DB exists on disk first? BIDSLayout(database_path=var)? where is this saved?
    ## should this be made / updated as part of BIDS-ification of dicoms?

    ## pull every t1w / dwi file name from BIDS layout
    if bidf:
        anat_files = layout.get(extension='.nii.gz', **bidf['t1w'])
        dmri_files = layout.get(extension='.nii.gz', **bidf['dwi'])
    else:
        anat_files = layout.get(suffix='T1w', extension='.nii.gz')
        dmri_files = layout.get(suffix='dwi', extension='.nii.gz')

    ## preallocate candidate anatomical files
    canat = []

    logger.info("Parsing Anatomical Files...")
    for idx, anat in enumerate(anat_files):

        ## pull the data
        tmeta = anat.get_metadata()
        tvol = anat.get_image()

        ## what features of a T1 are most important to log when the goal is process w/ dMRI?
        ## Smaller / Larger image dim? Compute a basic QC/SNR feature?

        ## because PPMI doesn't have full sidecars, pull stuff

        try:
            tmcmode = tmeta['MatrixCoilMode']
        except:
            tmcmode = 'unknown'

        try:
            torient = tmeta['ImageOrientationText']
        except:
            torient = 'sag'

        try:
            tprotocol = tmeta['ProtocolName']
        except:
            tprotocol = 'unknown'

        logger.info("- "*25)
        logger.info(anat.filename)
        logger.info(f"Scan Type: {tmcmode}")
        logger.info(f"Data Shape: {tvol.shape}")

        ## if sense is in the encoded header drop it
        if tmcmode.lower() == 'sense':
            continue

        ## if it's not a sagittal T1, it's probably not the main
        if not torient.lower() == 'sag':
            continue

        ## look for Neuromelanin type scan in name somewhere
        if ('neuromel' in tprotocol.lower()):
            continue
        
        ## heudiconv heuristics file has some fields that could be reused.
        ## how much effort are we supposed to spend generalizing parsing to other inputs?
        
        ## append the data if it passes all the skips
        canat.append(anat)

    logger.info("- "*25)

    ## error if nothing passes
    if len(canat) == 0:
        raise ValueError(f'No valid T1 anat file for {participant_id} in ses-{session_id}.')
        
    ## check how many candidates there are
    if len(canat) > 1:
        logger.info('Still have to pick one...')
        npart = [ len(x.get_entities()) for x in canat ]
        oanat = canat[np.argmin(npart)]
    else:
        oanat = canat[0]

    logger.info(f"Selected anat file: {oanat.filename}")
    logger.info("= "*25)
    
    ## preallocate candidate dmri inputs
    cdmri = []
    cbv = np.empty(len(dmri_files))
    cnv = np.empty(len(dmri_files))
    cpe = []
    
    logger.info("Parsing Diffusion Files...")
    for idx, dmri in enumerate(dmri_files):

        tmeta = dmri.get_metadata()
        tvol = dmri.get_image()

        try:
            tpedir = tmeta['PhaseEncodingDirection']
        except:
            raise ValueError("INCOMPLETE SIDECAR: ['PhaseEncodingDirection'] is not defined in sidecar. This is required to accurately parse the dMRI data.")
            
        logger.info("- "*25)
        logger.info(dmri.filename)
        logger.info(f"Encoding Direction: {tmeta['PhaseEncodingDirection']}")
        logger.info(f"Data Shape: {tvol.shape}")

        ## store phase encoding data
        cpe.append(tmeta['PhaseEncodingDirection'])
        
        ## store image dimension
        if len(tvol.shape) == 4:
            cnv[idx] = tvol.shape[-1]
        elif len(tvol.shape) == 3:
            cnv[idx] = 1
        else:
            raise ValueError('dMRI File: {dmri.filename} is not 3D/4D.')
            
        ## build paths to bvec / bval data
        tbvec = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri.filename.replace('.nii.gz', '.bvec')).joinpath()
        tbval = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri.filename.replace('.nii.gz', '.bval')).joinpath()
        #tbval = layout.get_bval(dmri)
        #tbvec = layout.get_bvec(dmri)
        
        ## if bvec / bval data exist
        if os.path.exists(tbvec) & os.path.exists(tbval):
            logger.info('BVEC / BVAL data exists for this file')
            cbv[idx] = 1
        else:
            logger.info('BVEC / BVAL data does not exist for this file')
            cbv[idx] = 0

        ## append to output (?)
        cdmri.append(dmri)
        
    logger.info("- "*25)
    
    ## if there's more than 1 candidate with bv* files
    if sum(cbv == 1) > 1:
        
        logger.info("Continue checks assuming 2 directed files...")

        dmrifs = []
        
        ## pull the full sequences
        for idx, x in enumerate(cbv):
            if x == 1:
                logger.info(f"File {idx+1}: {dmri_files[idx].filename}")
                dmrifs.append(dmri_files[idx])

        ## if each shell is in a separate file, that would need to be determined and fixed here.
                
        ## if there are more than 2, quit - bad input
        if len(dmrifs) > 2:
            raise ValueError('Too many candidate full sequences.')

        ## split out to separate files
        dmrifs1 = dmrifs[0]
        dmrifs2 = dmrifs[1]
        
        ## pull phase encoding direction
        dmrifs1pe = dmrifs1.get_metadata()['PhaseEncodingDirection']
        dmrifs2pe = dmrifs2.get_metadata()['PhaseEncodingDirection']

        ## get sequence lengths
        dmrifs1nv = dmrifs1.get_image().shape[3]
        dmrifs2nv = dmrifs2.get_image().shape[3]

        ## get the sequence bvals
        dmrifs1va = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs[0].filename.replace('.nii.gz', '.bval')).joinpath())
        dmrifs2va = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs[1].filename.replace('.nii.gz', '.bval')).joinpath())

        ## get the sequence bvecs
        dmrifs1ve = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs[0].filename.replace('.nii.gz', '.bvec')).joinpath())
        dmrifs2ve = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs[1].filename.replace('.nii.gz', '.bvec')).joinpath())

        ## get the number of directions
        dmrifs1wd = dmrifs1ve[:,dmrifs1va > 0]
        dmrifs2wd = dmrifs2ve[:,dmrifs2va > 0]
        
        ## if the phase encodings are the same axis
        if (dmrifs1pe[0] == dmrifs2pe[0]):

            logger.info(f'Phase encoding axis: {dmrifs1pe[0]}')

            ## if the phase encodings match exactly
            if (dmrifs1pe == dmrifs2pe):

                ## print log for surprising situation of matching files
                logger.info('Sequences are not reverse encoded and are identical.')
                logger.info('Was the phase encoding not flipped during acquisition or are these sequences longitudinal?')
                logger.info('Unsure how to parse. Ignoring shorter (or second) sequence.')

                ## pick the longer (or first) sequence to return
                if dmrifs1wd.shape[1] >= dmrifs2wd.shape[1]:
                    didx = dmri_files.index(dmrifs1)
                else:
                    didx = dmri_files.index(dmrifs2)
                    
                rpe_out = None

            ## they're the same axis in opposite orientations
            else:
                
                logger.info(f"Forward Phase Encoding: {dmrifs1pe}")
                logger.info(f"Reverse Phase Encoding: {dmrifs2pe}")
                ## if the sequences are the same length
                if (dmrifs1nv == dmrifs2nv):
                    
                    logger.info('N volumes match. Assuming mirrored sequences.')

                    ## verify that bvecs match
                    if np.allclose(dmrifs1wd, dmrifs2wd):
                        logger.info(' -- Verified weighted directions match.')
                        ## identity matching bvecs may be fragile - add a failover tolerance?
                        
                    else:
                        logger.info(' -- Weighted directions are different. Sequences do not match.')

                    ## pull the first as forward
                    didx = dmri_files.index(dmrifs1) 

                    ## pull the second as reverse
                    rpeimage = dmrifs2.get_image()

                    ## load image data
                    rpedata = rpeimage.get_fdata() 

                    ## load bval data
                    rpeb0s = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs2.filename.replace('.nii.gz', '.bval')).joinpath())

                    ## create average b0 data from sequence
                    rpeb0 = np.mean(rpedata[:,:,:,rpeb0s == 0], 3)

                    ## write to disk
                    tmp_dir = tempfile.mkdtemp() #TemoraryDirectory()
                    rpe_out = f'{participant_id}_rpe_b0.nii.gz'
                    rpe_data = nib.nifti1.Nifti1Image(rpeb0, rpeimage.affine)
                    rpe_shape = rpe_data.shape
                    nib.save(rpe_data, Path(tmp_dir, rpe_out).joinpath())

                else:

                    didx = dmri_files.index(dmrifs1)
                    
                    ## pull the second as reverse
                    rpeimage = dmrifs2.get_image()

                    ## load image data
                    rpedata = rpeimage.get_fdata() 

                    ## load bval data
                    rpeb0s = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs2.filename.replace('.nii.gz', '.bval')).joinpath())

                    ## create average b0 data from sequence
                    rpeb0 = np.mean(rpedata[:,:,:,rpeb0s == 0], 3)

                    ## write to disk
                    tmp_dir = tempfile.mkdtemp()
                    rpe_out = f'{participant_id}_rpe_b0.nii.gz'
                    rpe_data = nib.nifti1.Nifti1Image(rpeb0, rpeimage.affine)
                    rpe_shape = rpe_data.shape
                    nib.save(rpe_data, Path(tmp_dir, rpe_out).joinpath())

                    ## print a warning to log
                    logger.info('The reverse sequence has non-b0 directed volumes that cannot be used and will be ignored during processing.')
                    logger.info('If that is not the expected result, check that the data has been converted correctly.')
                
        else:

            raise ValueError(f'The phase encodings are on different axes: {dmrifs1pe}, {dmrifs2pe}\nCannot determine what to do.')
            
    else:
        
        logger.info("Continue checks assuming 1 directed file...")

        ## pull the index of the bvec that exists
        didx = np.argmax(cbv)
        
        ## pull phase encoding for directed volume
        fpe = cpe[didx]

        ## clean fpe of unnecessary + if it's there
        if fpe[-1] == "+":
            fpe = fpe[0]
        
        ## determine the reverse phase encoding
        if (len(fpe) == 1):
            rpe = fpe + "-"
        else:
            rpe = fpe[0]

        logger.info(f"Forward Phase Encoding: {fpe}")
        logger.info(f"Reverse Phase Encoding: {rpe}")
            
        ## look for the reverse phase encoded file in the other candidates
        if (rpe in cpe):
            
            rpeb0 = dmri_files[cpe.index(rpe)]
            rpevol = rpeb0.get_image()
            tmp_dir = tempfile.mkdtemp() #TemporaryDirectory()
            rpe_out = f'{participant_id}_rpe_b0.nii.gz'
            
            ## if the rpe file has multiple volumes
            if len(rpevol.shape) == 4:

                logger.info('An RPE file is present: Averaging b0 volumes to single RPE volume...')
                ## load and average the volumes
                rpedat = rpevol.get_fdata()
                rpedat = np.mean(rpedat, 3)

                ## and write the file to /tmp
                rpe_data = nib.nifti1.Nifti1Image(rpedat, rpevol.affine)
                rpe_shape = rpe_data.shape
                nib.save(rpe_data, Path(tmp_dir, rpe_out).joinpath())
                
            else:

                logger.info('An RPE file is present: Copying single the single RPE b0 volume...')
                ## otherwise, copy the input file to tmp
                rpe_shape = rpevol.shape
                shutil.copyfile(rpeb0, rpe_out)
                            
        else:
            
            logger.info("No RPE is found in candidate files")
            rpe_out = None
            
    logger.info("= "*25)
    
    ## default assignments
    dmrifile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename).joinpath()
    bvalfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename.replace('.nii.gz', '.bval')).joinpath()
    bvecfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename.replace('.nii.gz', '.bvec')).joinpath()
    anatfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'anat', oanat.filename).joinpath()

    ## condition empty path
    if rpe_out == None:
        rpe_file = None
    else:
        rpe_file = Path(tmp_dir, rpe_out).joinpath()

    ## set the phase encoding direction
    if ('i' in dmri_files[didx].get_metadata()['PhaseEncodingDirection']):
        phase = 'x'
    if ('j' in dmri_files[didx].get_metadata()['PhaseEncodingDirection']):
        phase = 'y'
    else:
        logger.info('An unlikely phase encoding has been selected.')
        phase = 'z'

    ## set the total readout time for topup
    readout = dmri_files[didx].get_metadata()['TotalReadoutTime']
    
    ## return the paths to the input files to copy
    return(dmrifile, bvalfile, bvecfile, anatfile, rpe_file, phase, readout)

def run(participant_id, global_configs, session_id, output_dir, use_bids_filter, dti_shells=None, fodf_shells=None, sh_order=None, logger=None):
    """ Runs TractoFlow command with Nextflow
    """

    ## extract the config options
    DATASET_ROOT = global_configs["DATASET_ROOT"]
    CONTAINER_STORE = global_configs["CONTAINER_STORE"]
    TRACTOFLOW_CONTAINER = global_configs["PROC_PIPELINES"]["tractoflow"]["CONTAINER"]
    TRACTOFLOW_VERSION = global_configs["PROC_PIPELINES"]["tractoflow"]["VERSION"]
    TRACTOFLOW_CONTAINER = TRACTOFLOW_CONTAINER.format(TRACTOFLOW_VERSION)
    SINGULARITY_TRACTOFLOW = f"{CONTAINER_STORE}/{TRACTOFLOW_CONTAINER}"
    LOGDIR = f"{DATASET_ROOT}/scratch/logs"
    SINGULARITY_COMMAND = global_configs["SINGULARITY_PATH"]
    
    ## initialize the logger
    if logger is None:
        log_file = f"{LOGDIR}/{participant_id}_ses-{session_id}_tractoflow.log"
        logger = my_logger.get_logger(log_file)

    ## log the info
    logger.info("-"*75)
    logger.info(f"Using DATASET_ROOT: {DATASET_ROOT}")
    logger.info(f"Using participant_id: {participant_id}, session_id: {session_id}")

    ## define default output_dir if it's not overwrote
    if output_dir is None:
        output_dir = f"{DATASET_ROOT}/derivatives"

    ## build paths to files
    bids_dir = f"{DATASET_ROOT}/bids"
    tractoflow_dir = f"{output_dir}/tractoflow/v{TRACTOFLOW_VERSION}"

    ## Copy bids_filter.json 
    if use_bids_filter:
        if not os.path.exists(f"{DATASET_ROOT}/proc/bids_filter_tractoflow.json"):
            logger.info(f"Copying ./bids_filter.json to {DATASET_ROOT}/proc/bids_filter_tractoflow.json (to be seen by Singularity container)")
            shutil.copyfile(f"{CWD}/bids_filter.json", f"{DATASET_ROOT}/proc/bids_filter_tractoflow.json")
        else:
            logger.info(f"Using found {DATASET_ROOT}/proc/bids_filter_tractoflow.json")
        
    ## build paths for outputs
    tractoflow_out_dir = f"{tractoflow_dir}/output/ses-{session_id}"
    tractoflow_home_dir = f"{tractoflow_out_dir}/{participant_id}"
    if not os.path.exists(Path(f"{tractoflow_home_dir}")):
        Path(f"{tractoflow_home_dir}").mkdir(parents=True, exist_ok=True)

    ## call the file parser to copy the correct files to the input structure
    dmrifile, bvalfile, bvecfile, anatfile, rpe_file, phase, readout = parse_data(global_configs, bids_dir, participant_id, session_id, use_bids_filter, logger)
    ## use_bids_filter may need to be path to the filter so it can be loaded, not the logical

    ## build paths to working inputs
    tractoflow_input_dir = f"{tractoflow_dir}/input"
    tractoflow_nxtf_inp = f"{tractoflow_input_dir}/{participant_id}_ses-{session_id}"
    tractoflow_subj_dir = f"{tractoflow_input_dir}/{participant_id}_ses-{session_id}/{participant_id}"
    tractoflow_work_dir = f"{tractoflow_dir}/work/{participant_id}_ses-{session_id}"
    nextflow_logdir = f"{LOGDIR}/nextflow"
    
    ## check if working values already exist
    if not os.path.exists(Path(f"{tractoflow_subj_dir}")):
        Path(f"{tractoflow_subj_dir}").mkdir(parents=True, exist_ok=True)

    ## build working directory if it doesn't exist already to preserve file times
    if not os.path.exists(Path(f"{tractoflow_work_dir}")):
        Path(f"{tractoflow_work_dir}").mkdir(parents=True, exist_ok=True)

    ## build path to nextflow folder in logs for storing .nextflow* files for each launch
    if not os.path.exists(Path(f"{nextflow_logdir}")):
        Path(f"{nextflow_logdir}").mkdir(parents=True, exist_ok=True)
        
    ## just make copies if they aren't already there - resume option cannot work w/ modified (recopied) files, so check first
    ## delete on success?
    if len(os.listdir(tractoflow_subj_dir)) == 0:
        shutil.copyfile(dmrifile, Path(tractoflow_subj_dir, 'dwi.nii.gz').joinpath())
        shutil.copyfile(bvalfile, Path(tractoflow_subj_dir, 'bval').joinpath())
        shutil.copyfile(bvecfile, Path(tractoflow_subj_dir, 'bvec').joinpath())
        shutil.copyfile(anatfile, Path(tractoflow_subj_dir, 't1.nii.gz').joinpath())
        if os.path.exists(rpe_file):
            shutil.copyfile(rpe_file, Path(tractoflow_subj_dir, 'rev_b0.nii.gz').joinpath())
    
    ## load the bval / bvec data
    bval = np.loadtxt(bvalfile)
    bvec = np.loadtxt(bvecfile)
    #sh_order = int(sh_order)

    #logger.info(f'bval: {bval}')
    
    ## round shells to get b0s that are ~50 / group shells that are off by +/- 10
    rval = bval + 49 ## this either does too much or not enough rounding for YLO dataset
    rval = np.floor(rval / 100) * 100
    rval = rval.astype(int) ## I don't know how to get around just overwriting with the "fix"
    np.savetxt(Path(tractoflow_subj_dir, 'bval').joinpath(), rval, fmt='%1.0i', newline=' ')
    
    ## pull the number of shells
    bunq = np.unique(rval)
    nshell = bunq.size - 1

    ## the default tensor shell
    ## where only the shell closest to 1000 is used
    #dten = rval[np.where(np.abs(rval - 1000) == np.min(np.abs(rval) - 1000))[0]]
    dten = rval[np.abs(rval-1000) == np.min(np.abs(rval - 1000))][0]
    
    if dti_shells is None:
        logger.info(f'No requested shell(s) passed to tensor fitting. Automatically extracting the shell closest to 1000.')
        dti_use = str(dten)
        logger.info(f'The tensor will be fit on data with b = {dti_use}')
    else:
        dti_use = dti_shells

    if fodf_shells is None:
        logger.info(f'No requested shell(s) passed for ODF fitting. Automatically using all non-zero shells.')
        odf_use = ','.join(map(str, np.unique(rval)[1:]))
        logger.info(f'The ODF will be fit on data with b = {odf_use}')
    else:
        odf_use = fodf_shells
        
    ## fix the passed bvals

    ## convert to integer
    dti_use = list(map(int, dti_use.split(",")))
    odf_use = list(map(int, odf_use.split(",")))

    ## create merged list of requested shells
    rshell = np.unique([ dti_use + odf_use ])
    logger.info(f'Requested shells: {rshell}')

    ## if any requested shell(s) are absent within data, error w/ useful warning
    mshell = np.setdiff1d(rshell, bunq)
    if mshell.size == 0:
        logger.info('Requested shells are available in the data.')
    else:
        logger.info(f'Requested shells are not present in the data: {mshell}')
        raise ValueError('Unable to process shell data not present in the data.')

    ## convert back to space separated strings to tractoflow can parse it
    dti_use = ' '.join(map(str, dti_use))
    odf_use = ' '.join(map(str, odf_use))

    ## pull lmax possible from all directions
    
    ## logical index of b0 values
    b0idx = rval == 0
        
    ## check that vectors are unique
    tvec = bvec[:,~b0idx]
    tdir = np.unique(tvec, axis=0)
    
    ## compute and print the maximum shell
    dlmax = max_lmax(tdir.shape[1])
    logger.info(f'The largest supported lmax is: {dlmax}')
    logger.info(f' -- The largest possible lmax is generally determined by the number of values in each shell.')
    
    if sh_order is None:
        sh_order = 6
        logger.info(f'No lmax is requested by user. Fitting highest lmax possible based on the data.')
        #logger.info(f' -- Fitting lmax: ({plmax})')
    else:
        if int(sh_order) <= dlmax:
            logger.info(f'Determining if data supports an max lmax of {sh_order}.')
        else:
            raise ValueError(f'Requested lmax of {sh_order} is higher than the data can support.')
        
    ## deal with multishell data
    if nshell == 1:
        plmax = dlmax
        logger.info(f"Single shell data has b = {int(bunq[1])}")
        logger.info(f" -- Shell b = {int(bunq[1])} has {tdir.shape[1]} directions capable of a max lmax: {plmax}.")

    ## have to check the utility of every shell
    else:

        logger.info(f"Multishell data has shells b = {bunq[1:]}")
        mlmax = []
        mldir = []
        for shell in bunq[1:]:

            ## pull the shells
            tndir = rval == shell
            #b0idx = rval == 0

            ## check that directed vectors are unique
            tvec = bvec[:,tndir]
            tdir = np.unique(tvec, axis=0)
            mldir.append(tdir.shape[1])
            
            ## compute and print the maximum lmax per shell
            tlmax = max_lmax(tdir.shape[1])
            mlmax.append(tlmax)
            logger.info(f" -- Shell b = {int(shell)} has {tdir.shape[1]} directions capable of a max lmax: {tlmax}")

        ## the max lmax within any 1 shell is used
        plmax = max(mlmax)
        logger.info(f"The maximum lmax for any one shell is: {plmax}")
        
    ## if lmax too large, reset with warning
    if int(sh_order) <= plmax:
        logger.info(f"Running model with lmax: {sh_order}")
    else:
        if int(sh_order) <= dlmax:
            logger.info(f'Running model with lmax: {sh_order}')
            logger.info(f' -- This lmax is in excess of what any one shell supports, but there are (theoretically) suffienct total directions.')
        else:
            logger.warning(f"The requested lmax ({sh_order}) exceeds the theoretical capabilities of the data ({dlmax})")
            logger.warning(f"Generally, you do not want to fit an lmax in excess of any one shell's ability in the data.")
            raise ValueError('The requested lmax is not supported by the data. This got past the first raise ValueError.')
            #logger.info(f"Running model with overrode lmax: {sh_order}")
            #sh_order = plmax

    ## hard coded inputs to the tractoflow command in mrproc
    profile='fully_reproducible'
    ncore=4

    ## drop sub- from participant ID
    tf_id = participant_id.replace('sub-', '')

    ## path to pipelines
    TRACTOFLOW_PIPE=f'{DATASET_ROOT}/workflow/proc_pipe/tractoflow'
    
    ## nextflow arguments for logging - this is fixed for every run
    NEXTFLOW_CMD=f"nextflow -log {LOGDIR}/{participant_id}_ses-{session_id}_nf-log.txt run /scilus_flows/tractoflow/main.nf -work-dir {tractoflow_work_dir} -with-trace {LOGDIR}/{participant_id}_ses-{session_id}_nf-trace.txt -with-report {LOGDIR}/{participant_id}_ses-{session_id}_nf-report.html"
    
    ## compose tractoflow arguments
    TRACTOFLOW_CMD=f""" --input {tractoflow_nxtf_inp} --output_dir {tractoflow_out_dir} --run_t1_denoising --run_gibbs_correction --encoding_direction {phase} --readout {readout} --dti_shells "0 {dti_use}" --fodf_shells "0 {odf_use}" --sh_order {sh_order} --profile {profile} --processes {ncore}"""

    ## TractoFlow arguments can be printed multiple ways that appear consistent with the documentation but are parsed incorrectly by nextflow.
    ## .nexflow.log (a run log that documents what is getting parsed by nexflow) shows additional quotes being added around the dti / fodf parameters. Something like: "'0' '1000'"
    ## Obviously, this breaks the calls that nextflow tries to make at somepoint (typically around Normalize_DWI) because half the command becomes an unfinished text block from the mysteriously added quotes.
    ## I don't know if the problem is python printing unhelpful/inaccurate text to the user or if nextflow can't parse its own input arguments correctly.
    
    ## add resume option if working directory is not empty
    if not len(os.listdir(tractoflow_work_dir)) == 0:
        TRACTOFLOW_CMD = TRACTOFLOW_CMD + " -resume"
    
    ## build command line call
    CMD_ARGS = NEXTFLOW_CMD + TRACTOFLOW_CMD 
    
    ## log what is called
    logger.info("-"*75)
    logger.info(f"Running TractoFlow for participant: {participant_id}")

    ## singularity 
    SINGULARITY_CMD=f"{SINGULARITY_COMMAND} exec --cleanenv -H {nextflow_logdir} -B {nextflow_logdir}:/nextflow -B {DATASET_ROOT} {SINGULARITY_TRACTOFLOW}"

    CMD=SINGULARITY_CMD + " " + CMD_ARGS
    logger.info("+"*75)
    logger.info(f"Command passed to system:\n\n{CMD}\n")
    logger.info("+"*75)
    
    ## there's probably a better way to try / catch the .run() call here
    try:
        logger.info('Attempting Run')
        #tractoflow_proc = subprocess.run(CMD, shell=True)
    except Exception as e:
        logger.error(f"TractoFlow run failed to launch with exception: {e}")

    logger.info('End of TractoFlow run script.')

if __name__ == '__main__':
    ## argparse
    HELPTEXT = """
    Script to run TractoFlow 
    """

    ## parse inputs
    parser = argparse.ArgumentParser(description=HELPTEXT)
    parser.add_argument('--global_config', type=str, help='path to global configs for a given nipoppy dataset', required=True)
    parser.add_argument('--participant_id', type=str, help='participant id', required=True)
    parser.add_argument('--session_id', type=str, help='session id for the participant', required=True)
    parser.add_argument('--output_dir', type=str, default=None, help='specify custom output dir (if None --> <DATASET_ROOT>/derivatives)')
    parser.add_argument('--use_bids_filter', action='store_true', help='use bids filter or not')
    parser.add_argument('--dti_shells', type=str, default=None, help='shell value(s) on which a tensor will be fit', required=False)
    parser.add_argument('--fodf_shells', type=str, default=None, help='shell value(s) on which the CSD will be fit', required=False)
    parser.add_argument('--sh_order', type=str, default=None, help='The order of the CSD function to fit', required=False)
    
    ## extract arguments
    args = parser.parse_args()
    global_config_file = args.global_config
    participant_id = args.participant_id
    session_id = args.session_id
    output_dir = args.output_dir # Needed on BIC (QPN) due to weird permissions issues with mkdir
    dti_shells=args.dti_shells
    fodf_shells=args.fodf_shells
    sh_order=args.sh_order
    use_bids_filter = args.use_bids_filter

    ## Read global config
    with open(global_config_file, 'r') as f:
        global_configs = json.load(f)

    ## make valid tractoflow call based on inputs    
    run(participant_id, global_configs, session_id, output_dir, use_bids_filter, dti_shells, fodf_shells, sh_order)
