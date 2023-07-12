import mne
import os

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)


def test_plot_seeg():
    if not os.path.exists(os.path.join(subjects_dir, "1")):
        os.symlink(os.path.join(subjects_dir, 'sample'),
                   os.path.join(subjects_dir, '1'))
    from ieeg.viz.mri import plot_subj
    raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")
    plot_subj(raw, subj_dir=subjects_dir, show=False)