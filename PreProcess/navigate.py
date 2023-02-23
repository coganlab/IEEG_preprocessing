import os.path as op
import re
from os import walk, listdir, mkdir
from typing import Union, List, Tuple, Dict

import mne
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile, parse_file_entities
from mne_bids import read_raw_bids, BIDSPath, write_raw_bids

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from PreProcess.timefreq.utils import to_samples, Signal  # noqa: E402
from PreProcess.utils.utils import PathLike, LAB_root  # noqa: E402

RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
    """Looks for the .dat file in a specified folder

    Parameters
    ----------
    folder : PathLike
        The folder to search in.

    Returns
    -------
    Tuple[PathLike, PathLike]
        The paths to the ieeg and cleanieeg files.
    """
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if re.match(r".*cleanieeg\.dat.*", file):
                cleanieeg: PathLike = op.join(root, file)
            elif re.match(r".*ieeg\.dat.*", file):
                ieeg: PathLike = op.join(root, file)
            if ieeg is not None and cleanieeg is not None:
                return ieeg, cleanieeg
    raise FileNotFoundError("Not all .dat files were found:")


def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    """Searches a BIDSLayout for a file and returns a BIDSPath to it.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    **kwargs : dict
        The parameters to search for. See BIDSFile.get() for more info.

    Returns
    -------
    BIDSPath
        The BIDSPath to the file.
    """
    my_search: List[BIDSFile] = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file, "
                                "try adding more search terms")
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    found = my_search[0]
    entities = found.get_entities()
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


def raw_from_layout(layout: BIDSLayout, preload: bool = False,
                    run: Union[list[int], int] = None, **kwargs) -> mne.io.Raw:
    """Searches a BIDSLayout for a raw file and returns a mne Raw object.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    run : Union[List[int], int], optional
        The run to search for, by default None
    preload: bool
        Whether to laod the data into memory or not, by default False
    **kwargs : dict
        The parameters to search for. See BIDSFile.get() for more info.

    Returns
    -------
    mne.io.Raw
    """
    if run is None:
        runs = layout.get(return_type="id", target="run", **kwargs)
    else:
        runs = list(run)
    raw: List[mne.io.Raw] = []
    if runs:
        for r in runs:
            BIDS_path = bidspath_from_layout(layout, run=r, **kwargs)
            new_raw = read_raw_bids(bids_path=BIDS_path)
            new_raw.load_data()
            raw.append(new_raw.copy())
            del new_raw
        whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    else:
        BIDS_path = bidspath_from_layout(layout, **kwargs)
        whole_raw = read_raw_bids(bids_path=BIDS_path)
    if preload:
        whole_raw.load_data()
    return whole_raw


def open_dat_file(file_path: str, channels: List[str],
                  sfreq: int = 2048, types: str = "seeg",
                  units: str = "uV") -> mne.io.RawArray:
    """Opens a .dat file and returns a mne.io.RawArray object.

    Parameters
    ----------
    file_path : str
        The path to the .dat file.
    channels : List[str]
        The channels to load.
    sfreq : int, optional
        The sampling frequency, by default 2048
    types : str, optional
        The channel types, by default "seeg"
    units : str, optional
        The units of the data, by default "uV"

    Returns
    -------
    mne.io.RawArray
    """
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    match units:
        case "V":
            factor = 1
        case "mV":
            factor = 1e-3
        case "uV":
            factor = 1e-6
        case "nV":
            factor = 1e-9
        case _:
            raise NotImplementedError("Unit " + units + " not implemented yet")
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array * factor, info)
    return raw


def get_data(sub_num: int = 53, task: str = "SentenceRep", run: int = None,
             BIDS_root: PathLike = None, lab_root=LAB_root):
    """Gets the data for a subject and task.

    Parameters
    ----------
    sub_num : int, optional
        The subject number, by default 53
    task : str, optional
        The task to get the data for, by default "SentenceRep"
    run : int, optional
        The run to get the data for, by default None
    BIDS_root : PathLike, optional
        The path to the BIDS directory, by default None
    lab_root : PathLike, optional
        The path to the lab directory, by default LAB_root

    Returns
    -------
    layout : BIDSLayout
        The BIDSLayout for the subject.
    raw : mne.io.Raw
        The raw data.
    D_dat_raw : mne.io.Raw
        The raw data from the D_Data folder.
    D_dat_filt : mne.io.Raw
        The filtered data from the D_Data folder.
    """
    for dir in listdir(lab_root):
        if re.match(r"BIDS-\d\.\d_" + task, dir) and "BIDS" in listdir(op.join(
                lab_root, dir)):
            BIDS_root = op.join(lab_root, dir, "BIDS")
            break
    if BIDS_root is None:
        raise FileNotFoundError("Could not find BIDS directory in {} for task "
                                "{}".format(lab_root, task))
    sub_pad = "D" + "{}".format(sub_num).zfill(4)
    subject = "D{}".format(sub_num)
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, run=run, subject=sub_pad, extension='.edf')
    D_dat_raw, D_dat_filt = find_dat(op.join(lab_root, "D_Data",
                                             task, subject))
    return layout, raw, D_dat_raw, D_dat_filt


def crop_data(raw: mne.io.Raw, start_pad: str = "10s", end_pad: str = "10s"
              ) -> mne.io.Raw:
    """Crops out long stretches of data with no events.

    Takes raw instance with annotated events and crops the instance so that the
    raw file starts at start_pad before the first event and stops an amount of
    time in seconds given by end_pad after the last event.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to crop.
    start_pad : str, optional
        The amount of time to pad the start of the file, by default "10s"
    end_pad : str, optional
        The amount of time to pad the end of the file, by default "10s"

    Returns
    -------
    mne.io.Raw
        The cropped raw file.
    """

    crop_list = []

    start_pad = to_samples(start_pad, raw.info['sfreq']) / raw.info['sfreq']
    end_pad = to_samples(end_pad, raw.info['sfreq']) / raw.info['sfreq']

    # split annotations into blocks
    annot = raw.annotations.copy()
    block_idx = [idx + 1 for idx, val in
                 enumerate(annot) if val['description'] == 'BAD boundary']
    block_annot = [annot[i: j] for i, j in
                   zip([0] + block_idx, block_idx +
                       ([len(annot)] if block_idx[-1] != len(annot) else []))]

    for block_an in block_annot:
        # remove boundary events from annotations
        no_bound = None
        for an in block_an:
            if 'boundary' not in an['description']:
                if no_bound is None:
                    no_bound = mne.Annotations(**an)
                else:
                    an.pop('orig_time')
                    no_bound.append(**an)

        # get start and stop time from raw.annotations onset attribute
        t_min = no_bound.onset[0] - start_pad
        t_max = no_bound.onset[-1] + end_pad

        # create new cropped raw file
        crop_list.append(raw.copy().crop(tmin=t_min, tmax=t_max))

    return mne.concatenate_raws(crop_list)


@mne.utils.verbose
def channel_outlier_marker(input_raw: Signal, outlier_sd: int = 3,
                           max_rounds: int = np.inf, verbose: bool = True
                           ) -> list[str]:
    """Identify bad channels by variance.

    Parameters
    ----------
    input_raw : Signal
        Raw data to be analyzed.
    outlier_sd : int, optional
        Number of standard deviations above the mean to be considered an
        outlier, by default 3
    max_rounds : int, optional
        Maximum number of varience estimations, by default runs until no
        more bad channels are found.
    verbose : bool, optional
        Print removed channels per estimation, by default True

    Returns
    -------
    list[str]
        List of bad channel names.
    """

    data = input_raw.get_data('data')  # (trials X) channels X time
    names = input_raw.copy().pick('data').ch_names
    bads = []  # output for bad channel names

    # Square the data and set zeros to small positive number
    R2 = np.square(data)
    R2[np.where(R2 == 0)] = 1e-9
    ch_dim = range(len(data.shape))[-2]  # dimension corresponding to channels

    # find all axes that are not channels (example: time, trials)
    axes = tuple(i for i in range(len(data.shape)) if not i == ch_dim)

    # Initialize stats loop
    sig = np.std(R2, axes)  # take standard deviation of each channel
    cutoff = (outlier_sd * np.std(sig)) + np.mean(sig)  # outlier cutoff
    i = 1

    # remove bad channels and re-calculate variance until no outliers are left
    while np.any(np.where(sig > cutoff)) and i <= max_rounds:

        # Pop out names to bads output using comprehension list
        [bads.append(names.pop(out-j)) for j, out in enumerate(
            np.where(sig > cutoff)[0])]

        # log channels excluded per round
        if verbose:
            mne.utils.logger.info(f'outlier round {i} channels: {bads}')

        # re-calculate per channel variance
        R2 = R2[..., np.where(sig < cutoff)[0], :]
        sig = np.std(R2, axes)
        cutoff = (outlier_sd * np.std(sig)) + np.mean(sig)
        i += 1

    return bads


def save_derivative(inst: Signal, layout: BIDSLayout, pipeline: str,
                    overwrite=False):
    """Save an intermediate data instance from a pipeline to a BIDS folder.

    Parameters
    ----------
    inst : Signal
        The data instance to save.
    layout : BIDSLayout
        The BIDSLayout of the original data.
    pipeline : str
        The name of the pipeline.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    """
    save_dir = op.join(layout.root, "derivatives", pipeline)
    if not op.isdir(save_dir):
        mkdir(save_dir)
    bounds = inst.annotations.copy()
    bounds = bounds[np.where(bounds.description == 'BAD boundary')[0]]
    bounds = [0] + list(bounds.onset) + [inst.times[-1]]
    for i, file in enumerate(inst.filenames):
        entities = parse_file_entities(file)
        entities['description'] = pipeline
        bids_path = BIDSPath(**entities, root=save_dir)
        run = inst.copy().crop(tmin=bounds[i], tmax=bounds[i+1])
        write_raw_bids(run, bids_path, allow_preload=True, format='EDF',
                       acpc_aligned=True, overwrite=overwrite)


if __name__ == "__main__":
    # %% Set up logging
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    TASK = "SentenceRep"
    sub_num = 29
    # layout, raw, D_dat_raw, D_dat_filt = get_data(sub_num, TASK)
    bids_root = LAB_root + "/BIDS-1.0_SentenceRep/BIDS"
    layout = BIDSLayout(bids_root)
    filt = mne.io.read_raw_fif(layout.root + "/derivatives/sub-D00" + str(
        sub_num) + "_" + TASK + "_filt_ieeg.fif")
    events, event_id = mne.events_from_annotations(filt)
    auds = mne.Epochs(filt, events, event_id['Audio'], baseline=None, tmin=-2,
                      tmax=5, preload=True, detrend=1)
    bads = channel_outlier_marker(auds)
    auds.info['bads'] = bads
