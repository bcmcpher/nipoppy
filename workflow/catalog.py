import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import argparse
import json


def read_manifest(manifest_csv, session_id, logger):
    # read current participant manifest 
    manifest_df = pd.read_csv(manifest_csv)

    # filter session
    manifest_df["session_id"] = manifest_df["session_id"].astype(str)
    manifest_df = manifest_df[manifest_df["session_id"] == session_id]
    
    manifest_df["participant_id"] = manifest_df["participant_id"].astype(str)
    participants = manifest_df["participant_id"].str.strip().values

    # generate dicom_id
    manifest_df["dicom_id"] = [''.join(filter(str.isalnum, idx)) for idx in participants]

    # generate bids_id
    manifest_df["bids_id"] = "sub-" + manifest_df["dicom_id"].astype(str)

    # check dicom files
    if "dicom_file" in manifest_df.columns:
        logger.info("Using dicom filename from the manifest.csv") 
    else:
        logger.warning("dicom_file is not specified in the manifest.csv")
        logger.info("Assuming dicom_id is the dicom filename") 
        manifest_df["dicom_file"] = manifest_df["dicom_id"].copy()

    return manifest_df

def list_dicoms(dcm_dir, logger):
    # check current dicom dir
    if Path.is_dir(Path(dcm_dir)):
        participant_dcm_dirs = next(os.walk(dcm_dir))[1]
    else:
        participant_dcm_dirs = []
        logger.warning(f"No participant dicoms found in {dcm_dir}")

    return participant_dcm_dirs

def list_bids(bids_dir, session_id, logger):
    # available participant bids dirs (for particular session)
    if Path.is_dir(Path(bids_dir)):
        current_bids_dirs = next(os.walk(bids_dir))[1]
        current_bids_session_dirs = []
        for pbd in current_bids_dirs:
            ses_dir_path = Path(f"{bids_dir}/{pbd}/ses-{session_id}")
            if Path.is_dir(ses_dir_path):
                current_bids_session_dirs.append(pbd)
    else:
        current_bids_session_dirs = []
        logger.warning(f"No participant bids dir found in {bids_dir}")

    return current_bids_session_dirs


def get_new_downloads(manifest_csv, raw_dicom_dir, session_id, logger):
    """ Identify new dicoms not yet inside <DATASET_ROOT>/scratch/raw_dicom
    """
    manifest_df = read_manifest(manifest_csv, session_id, logger)
    participants = set(manifest_df["participant_id"])
    n_participants = len(participants)

    # check raw dicom dir    
    available_raw_dicom_dirs = list_dicoms(raw_dicom_dir, logger)
    n_available_raw_dicom_dirs = len(available_raw_dicom_dirs)
    available_raw_dicom_dirs_participant_ids = list(manifest_df[manifest_df["dicom_file"].isin(available_raw_dicom_dirs)]["participant_id"].astype(str).values)

    # check mismatch between manifest and raw_dicoms
    download_dicom_dir_participant_ids = set(participants) - set(available_raw_dicom_dirs_participant_ids)
    n_download_dicom_dirs = len(download_dicom_dir_participant_ids)

    download_df = manifest_df[manifest_df["participant_id"].isin(download_dicom_dir_participant_ids)]

    logger.info("-"*50)
    logger.info(f"Identifying participants to be downloaded\n\n \
    - n_particitpants (listed in the mr_proc_manifest): {n_participants}\n \
    - n_available_raw_dicom_dirs: {n_available_raw_dicom_dirs}\n \
    - n_download_dicom_dirs: {n_download_dicom_dirs}\n")
    logger.info("-"*50)

    return download_df

def get_new_raw_dicoms(manifest_csv, raw_dicom_dir, dicom_dir, session_id, logger):
    """ Identify new raw_dicoms not yet reorganized inside <DATASET_ROOT>/dicom
    """
    manifest_df = read_manifest(manifest_csv, session_id, logger)
    participants = set(manifest_df["participant_id"])
    n_participants = len(participants)

    # check current dicom dir
    current_dicom_dirs = list_dicoms(dicom_dir, logger)
    n_participant_dicom_dirs = len(current_dicom_dirs)
    current_dicom_dirs_participant_ids = set(manifest_df[manifest_df["dicom_id"].isin(current_dicom_dirs)]["participant_id"].values)

    # check raw dicom dir    
    available_raw_dicom_dirs = list_dicoms(raw_dicom_dir, logger)
    n_available_raw_dicom_dirs = len(available_raw_dicom_dirs)
    available_raw_dicom_dirs_participant_ids = list(manifest_df[manifest_df["dicom_file"].isin(available_raw_dicom_dirs)]["participant_id"].astype(str).values)

    # check mismatch between manifest and raw_dicoms
    missing_dicom_dir_participant_ids = set(participants) - set(available_raw_dicom_dirs_participant_ids)
    n_missing_dicom_dirs = len(missing_dicom_dir_participant_ids)

    # identify participants to be reorganized   
    dicom_reorg_participants = set(participants) - current_dicom_dirs_participant_ids - missing_dicom_dir_participant_ids
    n_dicom_reorg_participants = len(dicom_reorg_participants)

    reorg_df = manifest_df[manifest_df["participant_id"].isin(dicom_reorg_participants)]

    logger.info("-"*50)
    logger.info(f"Identifying participants to be reorganized\n\n \
    - n_particitpants (listed in the mr_proc_manifest): {n_participants}\n \
    - n_particitpant_dicom_dirs (current): {n_participant_dicom_dirs}\n \
    - n_available_dicom_dirs: {n_available_raw_dicom_dirs}\n \
    - n_missing_dicom_dirs: {n_missing_dicom_dirs}\n \
    - dicom_reorg_participants: {n_dicom_reorg_participants}\n")
    logger.info("-"*50)

    return reorg_df

def get_new_dicoms(manifest_csv, dicom_dir, bids_dir, session_id, logger):
    """ Identify new dicoms not yet BIDSified
    """
    manifest_df = read_manifest(manifest_csv, session_id, logger)
    participants = set(manifest_df["participant_id"])
    n_participants = len(participants)
    dicom_ids = set(manifest_df["dicom_id"])
    bids_ids = set(manifest_df["bids_id"])

    # check current bids dir
    current_bids_dirs = list_bids(bids_dir, session_id, logger)
    current_bids_dirs = bids_ids & set(current_bids_dirs)
    n_current_bids_dirs = len(current_bids_dirs)

    # check current dicom dir
    available_dicom_dirs = list_dicoms(dicom_dir, logger)
    n_available_dicom_dirs = len(available_dicom_dirs)

    # check mismatch between manifest and participant dicoms
    missing_dicom_dirs = set(dicom_ids) - set(available_dicom_dirs)
    n_missing_dicom_dirs = len(missing_dicom_dirs)

    current_bids_dirs_dicom_ids = manifest_df[manifest_df["bids_id"].isin(current_bids_dirs)]["dicom_id"]

    # participants to process with Heudiconv
    heudiconv_participants = set(dicom_ids) - set(missing_dicom_dirs) - set(current_bids_dirs_dicom_ids)
    n_heudiconv_participants = len(heudiconv_participants)
    heudiconv_df = manifest_df[manifest_df["dicom_id"].isin(heudiconv_participants)]

    logger.info("-"*50)
    logger.info(f"Identifying participants to be BIDSified\n\n \
    - n_particitpants (listed in the mr_proc_manifest): {n_participants}\n \
    - n_current_bids_dirs (current): {n_current_bids_dirs}\n \
    - n_available_dicom_dirs (available): {n_available_dicom_dirs}\n \
    - n_missing_dicom_dirs: {n_missing_dicom_dirs}\n \
    - heudiconv participants to processes: {n_heudiconv_participants}\n")
    logger.info("-"*50)

    return heudiconv_df