import os.path as op
from collections import OrderedDict
import csv

import matplotlib
from mne.viz import Brain

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np

from ieeg import Signal, PathLike


def plot_overlay(image: nib.Nifti1Image, compare: nib.Nifti1Image,
                 title: str, thresh: float = None):
    """Plots an overlay of two images

    Parameters
    ----------
    image : nib.Nifti1Image
        The image to plot
    compare : nib.Nifti1Image
        The image to overlay
    title : str
        The title of the plot
    thresh : float, optional
        The threshold to apply to the overlay, by default None
    """
    image = nib.orientations.apply_orientation(
        np.asarray(image.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(image.affine))).astype(np.float32)
    compare = nib.orientations.apply_orientation(
        np.asarray(compare.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(compare.affine))).astype(np.float32)
    if thresh is not None:
        compare[compare < np.quantile(compare, thresh)] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(np.take(image, [image.shape[i] // 2], axis=i).squeeze().T,
                  cmap='gray')
        ax.imshow(np.take(compare, [compare.shape[i] // 2],
                          axis=i).squeeze().T, cmap='gist_heat', alpha=0.5)
        ax.invert_yaxis()
        ax.axis('off')
    fig.tight_layout()


def allign_CT(t1_path: PathLike, ct_path: PathLike, reg_affine=None
              ) -> nib.spatialimages.SpatialImage:
    """Alligns a CT scan to a T1 scan

    Parameters
    ----------
    t1_path : PathLike
        The path to the T1 scan
    ct_path : PathLike
        The path to the CT scan
    reg_affine : np.ndarray, optional
        The affine to use for registration, by default None

    Returns
    -------
    nib.spatialimages.SpatialImage
        The alligned CT scan
    """
    T1 = nib.load(t1_path)
    CT_orig = nib.load(ct_path)
    sdr_morph = None
    if reg_affine is None:
        reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
            CT_orig, T1, pipeline='all')
    CT_aligned = mne.transforms.apply_volume_registration(CT_orig, T1,
                                                          reg_affine,
                                                          sdr_morph)
    return CT_aligned


def show_brain(my_raw: Signal, trans: mne.transforms.Transform,
               sub_id: PathLike, subjects_dir: PathLike = None,
               overwrite: bool = False):
    """Shows the brain with the electrodes projected onto it

    Parameters
    ----------
    my_raw : Signal
        The data to plot
    trans : mne.transforms.Transform
        The transformation to use
    sub_id : PathLike
        The subject id
    subjects_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    overwrite : bool, optional
        Whether to overwrite the watershed bem, by default False
    """
    subjects_dir = get_sub_dir(subjects_dir)
    try:
        mne.bem.make_watershed_bem(sub_id, subjects_dir, overwrite=overwrite)
    except RuntimeError:
        pass
    my_raw.info = mne.preprocessing.ieeg.project_sensors_onto_brain(
        my_raw.info, trans, sub_id, subjects_dir=subjects_dir)
    brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white')
    brain = mne.viz.Brain(sub_id, surf='white', subjects_dir=subjects_dir,
                          title='Projection', **brain_kwargs)
    brain.add_sensors(my_raw.info, trans=trans)
    view_kwargs = dict(azimuth=60, elevation=100, distance=350,
                       focalpoint=(0, 0, -15))
    brain.show_view(**view_kwargs)


def head_to_mni(inst: Signal, sub: str, subj_dir: PathLike = None):
    """Transforms the head coordinates to MNI Talairach coordinates

    Parameters
    ----------
    inst : Union[mne.io.Raw, mne.Epochs, mne.Evoked]
        The data to transform
    sub : str
        The subject id
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    """
    subj_dir = get_sub_dir(subj_dir)
    montage = inst.get_montage()

    # first we need a head to mri transform since the data is stored in "head"
    # coordinates, let's load the mri to head transform and invert it
    head_mri_t = mne.coreg.estimate_head_mri_t(sub, subj_dir)
    # apply the transform to our montage
    montage.apply_trans(head_mri_t)

    montage.add_estimated_fiducials(sub, subj_dir)
    # now let's load our Talairach transform and apply it
    mri_mni_t = mne.read_talxfm(sub, subj_dir)
    montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

    # for fsaverage, "mri" and "mni_tal" are equivalent and, since
    # we want to plot in fsaverage "mri" space, we need use an identity
    # transform to equate these coordinate frames
    montage.apply_trans(
        mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))
    # montage.add_mni_fiducials(sub, subj_dir)
    inst.set_montage(montage)


def get_sub_dir(subj_dir: PathLike = None):
    """Gets the subjects directory

    Parameters
    ----------
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'

    Returns
    -------
    PathLike
        The subjects directory
    """
    if subj_dir is None:
        from os import path
        HOME = path.expanduser("~")
        LAB_root = path.join(HOME, "Box", "CoganLab")
        subj_dir = op.join(HOME, "Box", "ECoG_Recon")
    return subj_dir


def plot_gamma(evoked: mne.Evoked, subjects_dir: PathLike = None, **kwargs):
    """Plots the gamma power over time

    Parameters
    ----------
    evoked : mne.Evoked
        The data to plot
    subjects_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    **kwargs
        Additional arguments to pass to plot_on_average
        """
    data = evoked.copy().filter(30, 150).apply_hilbert(envelope=True)._data
    fig = plot_on_average(evoked.info, subjects_dir=subjects_dir, **kwargs)
    mne.viz.set_3d_view(fig, azimuth=0, elevation=70)

    xy, im = mne.viz.snapshot_brain_montage(fig, evoked.info)
    # convert from a dictionary to array to plot
    xy_pts = np.vstack([xy[ch] for ch in evoked.info['ch_names']])

    # get a colormap to color nearby points similar colors
    cmap = matplotlib.cm.get_cmap('viridis')

    # create the figure of the brain with the electrode positions
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Gamma power over time', size='large')
    ax.imshow(im)
    ax.set_axis_off()

    # normalize gamma power for plotting
    gamma_power = -100 * data / data.max()
    # add the time course overlaid on the positions
    x_line = np.linspace(-0.025 * im.shape[0], 0.025 * im.shape[0],
                         data.shape[1])
    for i, pos in enumerate(xy_pts):
        x, y = pos
        color = cmap(i / xy_pts.shape[0])
        ax.plot(x_line + x, gamma_power[i] + y, linewidth=0.5, color=color)


def plot_on_average(sigs: Signal | str | list[Signal | str],
                    subj_dir: PathLike = None, rm_wm: bool = True,
                    picks: list[int | str] = None, surface: str = 'pial'
                    ) -> matplotlib.figure.Figure:
    """Plots the signal on the average brain

    Takes a signal instance or list of signal instances and plots them on the
    fsaverage brain.


    Parameters
    ----------
    sigs : Union[Signal, list[Signal]]
        The signal(s) to plot
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon'
    rm_wm : bool, optional
        Whether to remove white matter electrodes, by default True
    picks : list[int | str], optional
        The channels to plot, by default None
    surface : str, optional
        The surface to plot on, by default 'pial'

    Returns
    -------
    matplotlib.figure.Figure
        The figure
    """

    subj_dir = get_sub_dir(subj_dir)
    brain = Brain('fsaverage', subjects_dir=subj_dir, cortex='low_contrast',
                  alpha=0.6, background='grey', surf=surface)

    if isinstance(sigs, Signal):
        sigs = [sigs]
    if isinstance(sigs, list):
        sigs = {get_sub(v): v for v in sigs}

    for subj, inst in sigs.items():

        new = inst.copy()

        to_fsaverage = mne.read_talxfm(subj, subj_dir)
        to_fsaverage = mne.transforms.Transform(fro='head', to='mri', trans=to_fsaverage['trans'])

        if picks is None:
            picks = new.ch_names

        if rm_wm:
            picks = pick_no_wm(picks, gen_labels(new, sub, subj_dir, picks))

        # Convert picks to indices
        these_picks = list(mne.pick_channels(new.ch_names, picks))

        if len(these_picks) == 0:
            continue

        # plot the data
        plot_subj(new, subj_dir, picks, True, 8, brain, to_fsaverage)

    return brain


def pick_no_wm(picks: list[str], labels: OrderedDict[str, list[str]]):
    """Picks the channels that are not in the white matter

    Parameters
    ----------
    picks : list[str | int]
        The channels to pick from
    labels : dict[str | int, list[str]]
        The labels for each channel

    Returns
    -------
    list[str | int]
        The channels that are not in the white matter
    """

    # remove 'Unknown' values from label lists
    for k, v in labels.items():
        while 'Unknown' in v:
            labels[k].remove('Unknown')
            v = labels[k]

    # remove corresponding picks with either 'White-Matter' in the left most entry or empty lists
    picks = [p for p in picks if labels[p] != []]
    picks = [p for p in picks if 'White-Matter' not in labels[p][0]]
    return picks


def get_sub(inst: Signal | mne.Info) -> str:
    """Gets the subject from the instance

    Parameters
    ----------
    inst : Signal
        The instance to get the subject from

    Returns
    -------
    str
        The subject"""
    if isinstance(inst, Signal):
        inst = inst.info
    return "D" + str(int(inst['subject_info']['his_id'][5:]))


def plot_subj(inst: Signal | mne.Info | str, subj_dir: PathLike = None,
              picks: list[str | int] = None, no_wm: bool = False,
              labels_every: int = 8, fig: Brain = None,
              trans=None) -> Brain:
    """Plots the electrodes on the subject's brain

    Parameters
    ----------
    inst : Signal | mne.Info | str
        The subject to plot
    subj_dir : PathLike, optional
        The subjects directory, by default HOME / 'Box' / 'ECoG_Recon'
    picks : list[str | int], optional
        The channels to plot, by default all
    no_wm : bool, optional
        Whether to remove the white matter channels, by default False
    labels_every : int, optional
        How often to label the channels, by default 8
    fig : Brain, optional
        The figure to plot on, by default None
    trans: mne.transforms.Transform, optional
        The transformation to apply, by default None

    Returns
    -------
    Brain
        The brain plot
    """
    if isinstance(inst, Signal):
        info = inst.info
        sub = get_sub(info)
    elif isinstance(inst, mne.Info):
        info = inst
        sub = get_sub(info)
    elif isinstance(inst, str):
        info = subject_to_info(inst, subj_dir)
        sub = inst
    else:
        raise TypeError(f"inst must be Signal, mne.Info, or str, not {type(inst)}")

    if subj_dir is None:
        subj_dir = get_sub_dir(subj_dir)
    if trans is None:
        trans = mne.transforms.Transform(fro='head', to='mri')
    if fig is None:
        fig = Brain(sub, subjects_dir=subj_dir, cortex='low_contrast', alpha=0.5,
                    background='grey', surf='pial')
    if picks is None:
        picks = info.ch_names
    if no_wm:
        picks = pick_no_wm(picks, gen_labels(info, sub, subj_dir, picks))

    pick_ind = mne.pick_channels(info.ch_names, picks)
    info: mne.Info = mne.pick_info(info, pick_ind)

    fig.add_sensors(info, trans)
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    pos = montage.get_positions()['ch_pos']

    names = picks[slice(0, info['nchan'], labels_every)]
    positions = np.array([pos[name] for name in names]) * 1000
    fig.plotter.add_point_labels(positions, names, shape=None)
    return fig


def subject_to_info(subject: str, subjects_dir: PathLike = None,
                    ch_types: str = "seeg") -> mne.Info:
    """Gets the info for a subject from the subjects directory

    Parameters
    ----------
    subject : str
        The subject to get the info for
    subjects_dir : PathLike, optional
        The subjects directory, by default HOME / 'Box' / 'ECoG_Recon'
    ch_types : str, optional
        The channel type, by default "seeg"

    Returns
    -------
    mne.Info
        The info for the subject
    """
    subjects_dir = get_sub_dir(subjects_dir)
    elec_file = op.join(subjects_dir, subject, 'elec_recon',
                        subject + '_elec_locations_RAS_brainshifted.txt')
    elecs = dict()
    with open(elec_file, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            line = row[0].split(" ")
            elecs["".join(line[0:2])] = tuple(float(n) / 1000 for n in line[2:5])
    info = mne.create_info(list(elecs.keys()), 2000, ch_types)
    montage = mne.channels.make_dig_montage(elecs, nasion=(0, 0, 0), coord_frame='ras')
    info.set_montage(montage)
    return info


def force2frame(montage: mne.channels.DigMontage, frame: str = 'mri'):
    """Forces the montage to be in the specified frame

    Parameters
    ----------
    montage : mne.channels.DigMontage
        The montage to force
    frame : str, optional
        The frame to force to, by default 'mri'
    """

    settings = dict(fro=montage.get_positions()['coord_frame'],
                    to=frame, trans=np.eye(4))
    # current subjects are in 'mri' space, even though it says head
    if not settings['fro'] == frame:
        trans = mne.transforms.Transform(**settings)
        montage.apply_trans(trans)


def gen_labels(info: mne.Info, sub: str = None, subj_dir: PathLike = None,
               picks: list[str | int] = None) -> OrderedDict[str, list[str]]:
    """Generates the labels for the electrodes

    Parameters
    ----------
    inst : Signal | PathLike
        The subject to get the labels for
    sub : str, optional
        The subject name, by default None
    subj_dir : PathLike, optional
        The subjects directory, by default None
    picks : list[str | int], optional
        The channels to plot, by default None

    Returns
    -------
    dict[str, list]
        The labels for the electrodes
    """

    sub = get_sub(info) if sub is None else sub
    subj_dir = get_sub_dir(subj_dir)
    montage = info.get_montage()
    force2frame(montage, 'mri')
    aseg = 'aparc.a2009s+aseg'  # parcellation/anatomical segmentation atlas
    labels, colors = mne.get_montage_volume_labels(
        montage, sub, subjects_dir=subj_dir, aseg=aseg)

    new_labels = OrderedDict()
    if picks is not None:
        for i, key in enumerate(labels.keys()):
            if any((i in picks, key in picks)):
                new_labels[key] = labels[key]

    return new_labels


if __name__ == "__main__":
    from ieeg.io import get_data, raw_from_layout
    from os import path
    HOME = path.expanduser("~")
    LAB_root = path.join(HOME, "Box", "CoganLab")
    # %% Set up logging
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    TASK = "SentenceRep"
    sub_num = 29
    layout = get_data(TASK, root=LAB_root)
    subj_dir = op.join(LAB_root, "ECoG_Recon_Full")
    sub_pad = "D" + str(sub_num).zfill(4)
    sub = "D{}".format(sub_num)

    filt = raw_from_layout(layout.derivatives['clean'], subject=sub_pad,
                       extension='.edf', desc='clean', preload=False)
    ##
    # rr, tris = mne.read_surface(op.join(get_sub_dir(), sub, 'surf', 'lh.pial'))
    # renderer = mne.viz.backends.renderer.create_3d_figure(
    #     size=(600, 600), bgcolor="w", scene=False
    # )
    # gray = (0.5, 0.5, 0.5)
    # renderer.mesh(rr[:,0], rr[:,1], rr[:,2], triangles=tris, color=gray)
    # view_kwargs = dict(elevation=90, azimuth=0)  # camera at +X with +Z up
    # mne.viz.set_3d_view(
    #     figure=renderer.figure, distance=350, focalpoint=(0.0, 0.0, 40.0),
    #     **view_kwargs
    # )
    # renderer.show()
    ##
    brain = plot_subj(filt)
    plot_on_average(filt)
    # plot_gamma(raw)
    # head_to_mni(raw, sub)
    # trans = mne.coreg.estimate_head_mri_t(sub, subj_dir)
    # mne.bem.make_watershed_bem(sub, subj_dir,
    # brainmask="../mri/brainmask.mgz")
    # fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=sub,
    #                              subjects_dir=subj_dir, dig=True,
    #                              show_axes=True)
    # # %%
    # T1_path = layout.get(return_type="path", subject=sub_pad,
    #                      extension="nii.gz")[0]
    # CT_path = T1_path.path.replace("T1w.nii.gz", "CT.nii.gz")
    # # filt = mne.io.read_raw_fif("D24_filt_ieeg.fif")
    # CT_aligned = allign_CT(T1_path, CT_path, sub)
    # subj_trans = mne.coreg.estimate_head_mri_t(sub, subjects_dir=subj_dir)
    # gui = mne.gui.locate_ieeg(raw.info, subj_trans, CT_aligned, subject=sub,
    #                           subjects_dir=subj_dir, verbose=10)
