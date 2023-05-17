import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from bids import BIDSLayout

def parse_data(bids_dir, participant_id, session_id, logger=None):
    """ Parse and verify the input files to build TractoFlow's simplified input to avoid their custom BIDS filter
    """

    ## because why parse subject ID the same as bids ID?
    subj = participant_id.replace('sub-', '')

    print('Building BIDS Layout...')
    
    ## parse directory
    layout = BIDSLayout(bids_dir)

    ## pull every t1w / dwi file name from BIDS layout
    anat_files = layout.get(subject=subj, session=session_id, suffix='T1w', extension='.nii.gz', return_type='object')
    dmri_files = layout.get(subject=subj, session=session_id, suffix='dwi', extension='.nii.gz', return_type='object')
        
    ## preallocate candidate anatomical files
    canat = []

    print("Parsing Anatomical Files...")
    for idx, anat in enumerate(anat_files):

        ## pull the data
        tmeta = anat.get_metadata()
        tvol = anat.get_image()

        print("- - - - - - - - - -")
        print(anat.filename)
        print(f"Scan Type: {tmeta['MatrixCoilMode']}\nData Shape: {tvol.shape}")

        ## if sense is in the encoded header drop it
        if tmeta['MatrixCoilMode'].lower() == 'sense':
            continue

        ## if it's not a sagittal T1, it's probably not the main
        if not tmeta['ImageOrientationText'].lower() == 'sag':
            continue

        ## look for Neuromelanin type scan in name somewhere
        if ('neuromel' in tmeta['ProtocolName'].lower()):
            continue
        
        ## heudiconv heuristics file has some fields that could be reused.
        ## how much effort are we supposed to spend generalizing parsing to other inputs?
        
        ## append the data if it passes all the skips
        canat.append(anat)

    print("- - - - - - - - - -")

    ## error if nothing passes
    if len(canat) == 0:
        error(f'No valid anat in {participant_id} for {session_id}.')
        
    ## check how many candidates there are
    if len(canat) > 1:
        print('Still have to pick one...')
        npart = [ len(x.get_entities()) for x in canat ]
        oanat = canat[np.argmin(npart)]
    else:
        oanat = canat[0]

    print(f"Selected anat file: {oanat.filename}")
    print("= = = = = = = = = =")
    
    ## preallocate candidate dmri inputs
    cdmri = []
    cbv = np.empty(len(dmri_files))
    cnv = np.empty(len(dmri_files))
    cpe = []
    
    print("Parsing Diffusion Files...")
    for idx, dmri in enumerate(dmri_files):

        tmeta = dmri.get_metadata()
        tvol = dmri.get_image()
        
        print("- - - - - - - - - -")
        print(dmri.filename)
        print(f"Encoding Direction: {tmeta['PhaseEncodingDirection']}\nData Shape: {tvol.shape}")

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

        ## if bvec / bval data exist
        if os.path.exists(tbvec) & os.path.exists(tbval):
            print('BVEC / BVAL data exists for this file')
            cbv[idx] = 1
        else:
            print('BVEC / BVAL data does not exist for this file')
            cbv[idx] = 0

        ## append to output (?)
        cdmri.append(dmri)
        
    print("- - - - - - - - - -")

    ## if there's more than 1 candidate with bv* files
    if sum(cbv == 1) > 1:
        
        print("Continue checks assuming 2 directed files...")

        dmrifs = []
        
        ## pull the full sequences
        for x in cbv:
            if x == 1:
                dmrifs.append(dmri_files[x])

        ## if there are more than 2, quit - bad input
        if len(dmrifs) > 2:
            raise ValueError('Too many candidate full sequences.')

        ## split out to separate files
        dmrifs1 = dmrifs[0]
        dmrifs2 = dmrifs[1]

        ## pull phase encoding direction
        dmrifs1pe = dmrifs1.get_metadata()['PhaseEncodingDirection']
        dmrifs2pe = dmrifs2.get_metadata()['PhaseEncodingDirection']

        ## if the phase encodings are the same axis
        if (dmrifs1pe[0] == dmrifs2pe[0]):

            print('Phase encoding axis: {dmrifs1pe[0]}')
            print(f"Foward Phase Encoding:  {dmrifs1pe}\nReverse Phase Encoding: {dmrifs2pe}")

            ## pull the number of volumes
            dmrifs1nv = dmrifs1.get_image().shape[3]
            dmrifs2nv = dmrifs1.get_image().shape[3]

            ## if the sequences are the same length
            if (dmrifs1nv == dmrifs2nv):

                print('N volumes match. Assuming mirrored sequences.')

                ## verify that directions match?
                
                ## pull the first as forward
                didx = dmri_files.index(dmrifs1) 

                ## pull the second as reverse
                rpeimage = dmrifs2.get_image()

                ## load image data
                rpedata = rpeimage.get_fdata() 

                ## load bval data
                rpeb0s = np.loadtxt(Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmrifs2.filename.replace('.nii.gz', '.bval')).joinpath())

                ## create average b0 from sequence
                rpeb0 = np.mean(rpedata[:,:,:,rpeb0s == 0], 3)

                ## write to disk
                rpe_out = f'/tmp/{participant_id}_rpe_b0.nii.gz'
                rpe_data = nib.nifti1.Nifti1Image(rpeb0, rpeimage.affine)
                rpe_shape = rpe_data.shape
                nib.save(rpe_data, rpe_out)

            else:

                raise ValueError('The number of volumes do not match. Cannot determine what to do.')

        else:

            raise ValueError(f'The phase encodings are on different axes: {dmrifs1pe}, {dmrifs2pe}\nCannot determine what to do.')
            
    else:
        
        print("Continue checks assuming 1 directed file...")

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

        print(f"Foward Phase Encoding:  {fpe}\nReverse Phase Encoding: {rpe}")
            
        ## look for the reverse phase encoded file in the other candidates
        if (rpe in cpe):
            
            rpeb0 = dmri_files[cpe.index(rpe)]
            rpevol = rpeb0.get_image()
            rpe_out = f'/tmp/{participant_id}_rpe_b0.nii.gz'
            
            ## if the rpe file has multiple volumes
            if len(rpevol.shape) == 4:

                print('An RPE file is present: Averaging b0 volumes to single RPE volume...')
                ## load and average the volumes
                rpedat = rpevol.get_fdata()
                rpedat = np.mean(rpedat, 3)

                ## and write the file to /tmp
                rpe_data = nib.nifti1.Nifti1Image(rpedat, rpevol.affine)
                rpe_shape = rpe_data.shape
                nib.save(rpe_data, rpe_out)
                
            else:

                print('An RPE file is present: Copying single the single RPE b0 volume...')
                ## otherwise, copy the input file to tmp
                rpe_shape = rpevol.shape
                shutil.copyfile(rpeb0, rpe_out)
                            
        else:
            
            print("No RPE is found in candidate files")
            rpe_out = None
            
    print("= = = = = = = = = =")
    
    ## default assignments
    dmrifile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename).joinpath()
    bvalfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename.replace('.nii.gz', '.bval')).joinpath()
    bvecfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'dwi', dmri_files[didx].filename.replace('.nii.gz', '.bvec')).joinpath()
    anatfile = Path(bids_dir, participant_id, 'ses-' + session_id, 'anat', oanat.filename).joinpath()
    rpe_file = Path(rpe_out)
        
    ## return the paths to the input files to copy
    return(dmrifile, bvalfile, bvecfile, anatfile, rpe_file)


if __name__ == '__main__':

    ## code in the arguments
    bids_dir='/data/origami/bcmcpher/mrproc-dev/bids'
    participant_id='sub-MNI0056D864854'
    session_id='01'

    ## create the outputs
    dmrifile, bvalfile, bvecfile, anatfile, rpe_file = parse_data(bids_dir, participant_id, session_id)
    print(f"dMRI: {dmrifile}\nbval: {bvalfile}\nbvec: {bvecfile}\nAnat: {anatfile}\nRPEv: {rpe_file}")
