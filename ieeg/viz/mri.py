import csv
import os.path as op
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import singledispatch

import mne
import nibabel as nib
import numpy as np
from mne.viz import Brain

from ieeg import PathLike, Signal
from ieeg.io import get_elec_volume_labels
from ieeg.viz import _qt_backend, parula

_qt_backend()

import matplotlib  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


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


def imshow_mri(data, img: nib.spatialimages.SpatialImage,
               vox: tuple[int, int, int], xyz: dict, suptitle: str = ""):
    """Show an MRI slice with a voxel annotated.

    Parameters
    ----------
    data : np.ndarray
        The data to plot
    img : nib.spatialimages.SpatialImage
        The image to plot
    vox : tuple[int, int, int]
        The voxel to annotate
    xyz : dict
        The xyz coordinates of the voxel
    suptitle : str, optional
        The title of the plot, by default ""

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        The figure and axes of the plot
    """
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(6, 6))
    codes = nib.orientations.aff2axcodes(img.affine)
    # Figure out the title based on the code of this axis
    ori_slice = dict(
        P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal",
        R="Sagittal"
    )
    ori_names = dict(
        P="posterior", A="anterior", I="inferior", S="superior", L="left",
        R="right"
    )
    title = ori_slice[codes[0]]
    ax.imshow(data[i], vmin=10, vmax=120, cmap="gray", origin="lower")
    ax.axvline(k, color="y")
    ax.axhline(j, color="y")
    for kind, coords in xyz.items():
        annotation = "{}: {}, {}, {} mm".format(kind,
                                                *np.round(coords).astype(int))
        text = ax.text(k, j, annotation, va="baseline", ha="right",
                       color=(1, 1, 0.7))
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ]
        )
    # reorient view so that RAS is always rightward and upward
    x_order = -1 if codes[2] in "LIP" else 1
    y_order = -1 if codes[1] in "LIP" else 1
    ax.set(
        xlim=[0, data.shape[2] - 1][::x_order],
        ylim=[0, data.shape[1] - 1][::y_order],
        xlabel=f"k ({ori_names[codes[2]]}+)",
        ylabel=f"j ({ori_names[codes[1]]}+)",
        title=f"{title} view: i={i} ({ori_names[codes[0]]}+)",
    )
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    return fig


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


def get_grey_matter(subjects: Sequence[str]) -> set[str]:
    """Gets the grey matter channels for a list of subjects

    Parameters
    ----------
    subjects : Sequence[str]
        The subjects to get the grey matter channels for

    Returns
    -------
    set[str]
        The grey matter channels
    """
    grey_matter = set()
    for i, subj in enumerate(subjects):
        info = subject_to_info(get_sub(subj))
        parcel_label = gen_labels(info, get_sub(subj))
        subj_grey_matter = pick_no_wm(info.ch_names, parcel_label)

        # get the indices of channels in info that are not in grey_matter
        grey_matter |= {f"{subj}-{ch}" for ch in info.ch_names
                        if ch in subj_grey_matter}
    return grey_matter


def plot_on_average(sigs: Signal | str | mne.Info | list[Signal | str, ...],
                    subj_dir: PathLike = None, rm_wm: bool = False,
                    picks: list[int | str, ...] = None, surface: str = 'pial',
                    hemi: str = 'split', color: matplotlib.colors = (1, 1, 1),
                    size: float = 0.35, fig: Brain = None,
                    label_every: int = None, background: str = 'white',
                    units: str = 'm', transparency: float = 0.6,
                    average: str = 'fsaverage', show: bool = True) -> Brain:
    """Plots the signal on the average brain

    Takes a signal instance or list of signal instances and plots them on the
    fsaverage brain.

    Parameters
    ----------
    sigs : Union[Signal, list[Signal]]
        The signal(s) to plot. If a list, all electrodes will be the same
        color. If a single signal, each electrode will be a different color.
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon'
    rm_wm : bool, optional
        Whether to remove white matter electrodes, by default True
    picks : list[int | str], optional
        The channels to plot, by default None
    surface : str, optional
        The surface to plot on, by default 'pial'
    hemi : str, optional
        The hemisphere to plot, by default 'split'
    color : matplotlib.colors, optional
        The color to plot, by default (1, 1, 1). If None, each electrode group
        will be a different color.
    size : float, optional
        The size of the markers, by default 0.35
    fig : Brain, optional
        The figure to plot on, by default None
    label_every : int, optional
        How often to label the channels, by default None
    background: str, optional
        Background color
    units: str, optional
        Units of the electrodes
    transparency: float, optional
        Transparency of the brain
    average: str, optional
        The average brain to plot on, by default 'fsaverage'
    show: bool, optional
        Whether to show the figure, by default True

    Returns
    -------
    Brain
        The figure brain object
    """

    subj_dir = get_sub_dir(subj_dir)
    if fig is None:
        fig = Brain(average, subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=transparency, background=background, surf=surface,
                    hemi=hemi, units=units, show=show)

    if isinstance(sigs, (Signal, mne.Info)):
        sigs = [sigs]
        color = None
    if isinstance(sigs, Iterable):
        sigs = {get_sub(v): v for v in sigs}

    default_c = parula.mat_colors.copy()
    for subj, inst in sigs.items():

        if isinstance(inst, mne.Info):
            new = inst.copy()
        elif isinstance(inst, Signal):
            new = inst.info.copy()
        elif isinstance(inst, str):
            new = subject_to_info(subj)
            new['subject_info'] = dict(his_id=f"sub-{inst}")
        else:
            raise TypeError(type(inst))

        to_fsaverage = mne.read_talxfm(subj, subj_dir)
        if average == 'fsaverage':
            trans = mne.transforms.Transform(fro='head', to='mri',
                                             trans=to_fsaverage['trans'])
        else:
            from_average = mne.read_talxfm(average, subj_dir)
            to_average = np.dot(np.linalg.inv(from_average['trans']),
                                to_fsaverage['trans'])
            trans = mne.transforms.Transform(fro='head', to='mri',
                                             trans=to_average)

        these_picks = range(len(new.ch_names))
        if isinstance(picks, Iterable):
            if len(picks) == 0:
                continue
            elif isinstance(picks[0], int):
                these_picks = [new.ch_names[pick] for pick in these_picks if
                               pick in picks]
                picks = [p - len(new.ch_names) for p in
                         picks[len(these_picks):]]
            elif isinstance(picks[0], str):
                if '-' in picks[0]:
                    these_picks = [s.split('-')[1] for s in picks if
                                   s.split('-')[0] in new[
                                       'subject_info']['his_id']]
                else:
                    these_picks = [s for s in picks if s in new.ch_names]
        elif picks is not None:
            raise TypeError(picks)

        if len(these_picks) == 0:
            continue

        if rm_wm:
            these_picks = pick_no_wm(these_picks, gen_labels(
                new, subj, subj_dir, picks=new.ch_names))

        if len(these_picks) == 0:
            continue

        # select colors
        if color is None and len(sigs) > 1:
            this_color = []
            p_int = [new.ch_names.index(p) for p in these_picks]
            groups = _group_channels(mne.pick_info(new, p_int))
            n_groups = len(set(groups.values()))
            while len(this_color) < n_groups:
                this_color += [default_c.pop(0)]
        else:
            this_color = color

        # plot the data
        plot_subj(new, subj_dir, these_picks, False, fig=fig,
                  trans=trans, color=this_color, size=size,
                  labels_every=label_every, hemi=hemi, background=background,
                  show=show)

    return fig


def pick_no_wm(picks: list[str], labels: OrderedDict[str: list[str]]) -> list:
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
    bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter')

    # remove corresponding picks with either 'White-Matter' in the left most
    # entry or empty lists
    if isinstance(picks[0], int):
        picks = [list(labels.keys())[p] for p in picks]
    picks = [p for p in picks if not any(w in labels[p] for w in bad_words)]
    return picks


def get_sub(inst: Signal | mne.Info | str) -> str:
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
    elif isinstance(inst, str):
        return f"{inst[0]}{int(inst[1:])}"
    out_str = inst['subject_info']['his_id'][4:]
    if len(out_str) == 1:
        return out_str
    return out_str[0] + str(int("".join(s for s in out_str if s.isdigit())))


# TODO: figure out why elec positions are only correct in meters not mm

# jim add code 2/1
def plot_subj_sig_and_nonsig(inst: Signal | mne.Info | str, subj_dir: PathLike = None,
              sig_picks: list[str | int] = None, nonsig_picks: list[str | int] = None,
              no_wm: bool = False, labels_every: int | None = 8, surface: str = 'pial',
              hemi: str = 'both', fig: Brain = None, trans=None,
              sig_color: tuple = (1, 1, 0), nonsig_color: tuple = (1, 1, 1),
              size: float = 0.35, show: bool = True, background: str = 'white',
              title: str = None, units: str = 'm', transparency: float = 0.5
              ) -> Brain:
    """Plots significant and non-significant electrodes on the subject's brain with different colors.

    Additional Parameters
    ---------------------
    sig_picks : list[str | int], optional
        The significant channels to plot.
    nonsig_picks : list[str | int], optional
        The non-significant channels to plot.
    sig_color : tuple, optional
        The color of significant electrodes.
    nonsig_color : tuple, optional
        The color of non-significant electrodes.
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
        raise TypeError(
            f"inst must be Signal, mne.Info, or str, not {type(inst)}")

    if subj_dir is None:
        subj_dir = get_sub_dir(subj_dir)
    if trans is None:
        trans = mne.transforms.Transform(fro='head', to='mri')
    if fig is None:
        fig = Brain(sub, subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=transparency, background=background, surf=surface,
                    hemi=hemi, show=show, units=units)

    # Set the title if provided
    if title is not None:
        mne.viz.set_3d_title(fig, title, size=40)
    
    # Plot non-significant electrodes
    if nonsig_picks:
        if isinstance(nonsig_picks[0], str):
            nonsig_picks = mne.pick_channels(info.ch_names, nonsig_picks)
        nonsig_info = mne.pick_info(info, nonsig_picks)
        _plot_electrodes(fig, nonsig_info, hemi, nonsig_color, size, trans)

    # Plot significant electrodes
    if sig_picks:
        if isinstance(sig_picks[0], str):
            sig_picks = mne.pick_channels(info.ch_names, sig_picks)
        sig_info = mne.pick_info(info, sig_picks)
        _plot_electrodes(fig, sig_info, hemi, sig_color, size, trans)

    return fig

def _plot_electrodes(fig, info, hemi, color, size, trans):
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    pos = {k: v for k, v in montage.get_positions()['ch_pos'].items()}

    left = {k: p for k, p in pos.items() if k.startswith('L')}
    right = {k: p for k, p in pos.items() if k.startswith('R')}

    if left and hemi != 'rh':
        _add_electrodes(fig, info, 'lh', np.vstack(list(left.values())), color, size)
    if right and hemi != 'lh':
        _add_electrodes(fig, info, 'rh', np.vstack(list(right.values())), color, size)


def plot_subj(inst: Signal | mne.Info | str, subj_dir: PathLike = None,
              picks: list[str | int] = None, no_wm: bool = False,
              labels_every: int | None = 8, surface: str = 'pial',
              hemi: str = 'both', fig: Brain = None,
              trans=None, color: matplotlib.colors = None,
              size: float = 0.35, show: bool = True, background: str = 'white',
              title: str = None, units: str = 'm', transparency: float = 0.5
              ) -> Brain:
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
    surface : str, optional
        The surface to plot, by default 'pial'
    hemi : str, optional
        The hemisphere to plot, by default 'split'
    trans: mne.transforms.Transform, optional
        The transformation to apply, by default None
    color : matplotlib.colors, optional
        The color of the electrodes, by default (1,1,1)
    size : float, optional
        The size of the electrodes, by default 0.35
    show : bool, optional
        Whether to show the figure, by default True
    background: str, optional
        Background color
    title : string, optional
        Title the plot
    units: str, optional
        Units of the electrodes
    transparency: float, optional
        Transparency of the brain

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
        raise TypeError(
            f"inst must be Signal, mne.Info, or str, not {type(inst)}")

    if subj_dir is None:
        subj_dir = get_sub_dir(subj_dir)
    if trans is None:
        trans = mne.transforms.Transform(fro='head', to='mri')
    if fig is None:
        fig = Brain(sub, subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=transparency, background=background, surf=surface,
                    hemi=hemi, show=show, units=units)

    if isinstance(color, str):
        color = matplotlib.colors.to_rgb(color)

    # Set the title if provided
    if title is not None:
        mne.viz.set_3d_title(fig, title, size=40)
    if picks is None:
        picks = info.ch_names
    if no_wm:
        picks = pick_no_wm(picks,
                           gen_labels(info, sub, subj_dir, info.ch_names))
    if isinstance(picks[0], str):
        picks = mne.pick_channels(info.ch_names, picks)

    info: mne.Info = mne.pick_info(info, picks)

    # fig.add_sensors(info, trans)
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    pos = {k: v for k, v in montage.get_positions()['ch_pos'].items()}

    # Default montage positions are in m, whereas plotting functions assume mm
    left = {}
    right = {}
    for k, v in pos.items():
        if k.startswith('L'):
            left[k] = v
        elif k.startswith('R'):
            right[k] = v
        elif v[0] < 0:
            left[k] = v
        else:
            right[k] = v

    if left and hemi != 'rh':
        _add_electrodes(fig, info, 'lh', np.vstack(list(left.values())),
                        color, size)
    if right and hemi != 'lh':
        _add_electrodes(fig, info, 'rh', np.vstack(list(right.values())),
                        color, size)

    if labels_every is not None:
        settings = dict(shape=None, always_visible=True,
                        text_color=fig._fg_color, bold=False)
        _add_labels(fig, info, sub, labels_every, hemi,
                    (left, right), **settings)

    return fig


def _add_electrodes(fig: mne.viz.Brain, info: mne.Info, hemi: str,
                    pos: np.ndarray, colors: matplotlib.colors = None,
                    size: float = 0.35):
    groups = _group_channels(info)
    n_groups = len(set(groups.values()))
    if colors is None:
        colors = parula.mat_colors[:n_groups]
    elif (not np.isscalar(colors) and len(colors) != n_groups) \
            or isinstance(colors, tuple):
        colors = [colors] * n_groups
    else:
        colors = list(colors)

    i = 0
    vals = list(groups.values())
    while i < n_groups:
        start, end = vals.index(i), len(vals) - vals[::-1].index(i)
        shank_pos = pos[start:end]
        fig.add_foci(shank_pos, hemi=hemi, color=colors[i],
                     scale_factor=size)
        i += 1


def _group_channels(info, groups: dict = None) -> dict:
    """Automatically find a group based on the name of the channel."""
    if groups is not None:
        for name in info.ch_names:
            if name not in groups.keys():
                raise ValueError(f"{name} not found in ``groups``")
        return groups

    i = 0
    groups = dict()
    base_names = dict()
    for name in info.ch_names:
        # strip all numbers from the name
        base_name = "".join(
            [
                letter
                for letter in name
                if not letter.isdigit() and letter != " "
            ]
        )
        if base_name in base_names:
            # look up group number by base name
            groups[name] = base_names[base_name]
        else:
            groups[name] = i
            base_names[base_name] = i
            i += 1
    return groups


def _add_labels(fig, info, sub, every, hemi, lr, **kwargs):
    if len(info.ch_names) < every:
        return
    names = info.ch_names[slice(every - 1, None, every)]
    if not hemi == 'both':
        for hems, pos in enumerate(lr):
            if (not pos) or \
                    (hemi == 'lh' and hems == 1) or \
                    (hemi == 'rh' and hems == 0):
                continue

            plt_names = filter(lambda x: x.startswith(['L', 'R'][hems]), names)
            plt_names = [f'{sub}-{n}' for n in plt_names]
            positions = np.array([pos[n.split("-")[1]] for n in plt_names])
            if hemi == 'split':
                fig.plotter.subplot(0, hems)
            fig.plotter.add_point_labels(positions, plt_names, **kwargs)
    else:
        pos = {}
        for hem in lr:
            if hem:
                pos.update(hem)
        plt_names = [f'{sub}-{n}' for n in names]
        positions = np.array([pos[name] for name in names])
        fig.plotter.add_point_labels(positions, plt_names, **kwargs)


def subject_to_info(subject: str, subjects_dir: PathLike = None,
                    ch_types: str = "seeg", sfreq: int = 2000) -> mne.Info:
    """Gets the info for a subject from the subjects directory

    Parameters
    ----------
    subject : str
        The subject to get the info for
    subjects_dir : PathLike, optional
        The subjects directory, by default HOME / 'Box' / 'ECoG_Recon'
    ch_types : str, optional
        The channel type, by default "seeg"
    sfreq : int, optional
        The sampling frequency, by default 2000

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
            elecs["".join(line[0:2])] = tuple(
                float(n) / 1000 for n in line[2:5])
    info = mne.create_info(list(elecs.keys()), sfreq, ch_types)
    montage = mne.channels.make_dig_montage(elecs, nasion=(0, 0, 0),
                                            coord_frame='ras')
    info.set_montage(montage)
    return info


@singledispatch
def force2frame(montage: mne.channels.DigMontage, frame: str):
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


@force2frame.register
def _(info: mne.Info, frame: str):
    montage = info.get_montage()
    force2frame(montage, frame)
    info.set_montage(montage)


def gen_labels(info: mne.Info, sub: str = None, subj_dir: str = None,
               atlas: str = ".a2009s", picks: list[str] = None
               ) -> OrderedDict[str, list[str]]:
    """Generates the labels for the electrodes

    Parameters
    ----------
    info : mne.Info
        The subject to get the labels for
    sub : str, optional
        The subject to get the labels for, by default None
    subj_dir : PathLike, optional
        The subjects directory, by default None
    picks : list[str | int], optional
        The channels to plot, by default None
    atlas : str, optional
        The atlas to use, by default ".a2009s"

    Returns
    -------
    dict[str, list]
        The labels for the electrodes
    """
    sub = get_sub(info) if sub is None else sub
    subj_dir = get_sub_dir(subj_dir)
    montage = info.get_montage()
    force2frame(montage, 'mri')
    # aseg = 'aparc.a2009s+aseg'  # parcellation/anatomical segmentation atlas
    labels = get_elec_volume_labels(sub, subj_dir, 10, atlas)
    new_labels = OrderedDict()
    if picks is None:
        picks = info.ch_names
    bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter')
    for p in picks:
        i = 2
        label = labels.T[p].T
        if label[0] not in bad_words:
            new_labels[p] = label[0]
            continue
        while not ((not any(w in label[i] for w in bad_words)) and
                   float(label[i + 1]) > 0.05):
            if (i + 2) >= len(label.T):  # end of labels
                i = 0
                break
            elif label[i + 2].isspace():  # empty label
                i = 0
                break
            else:
                i += 2
        new_labels[p] = label[i]
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
    subj_dir = op.join(LAB_root, "..", "ECoG_Recon")
    sub_pad = "D" + str(sub_num).zfill(4)
    info = subject_to_info(f"D{sub_num}", subj_dir)
    labels = gen_labels(info, sub=f"D{sub_num}", subj_dir=subj_dir,
                        atlas=".BN_atlas")

    # sub = "D{}".format(sub_num)

    # filt = raw_from_layout(layout.derivatives['clean'], subject=sub_pad,
    #                        extension='.edf', desc='clean', preload=False)

    ##
    # sample_path = mne.datasets.sample.data_path()
    # subjects_dir = sample_path / "subjects"

    # brain = plot_subj("D5")
    # fig = plot_on_average(["D24", "D81"], rm_wm=False, hemi='both',
    #                       transparency=0.4,
    #                       picks=list(range(28)) + list(range(52, 176)),
    #                       color=None,
    #                       average="D79", background=(0, 0.4, 0.5))
    # plot_gamma(raw)
    # plot_on_average(["D22", "D28", "D64"],
    #                 picks=["D22-LPIF4", "D28-LPIO7", "D64-LAI6"],
    #                 label_every=1, hemi='lh', rm_wm=False, color='red',
    #                 size=1)
