"""Constants for the omnifall package."""

HF_REPO_ID = "simplexsigil2/omnifall"

# OF-Syn video archive on HF Hub
SYN_VIDEO_ARCHIVE = "data_files/omnifall-synthetic_av1.tar"

# OOPS video download
OOPS_URL = "https://oops.cs.columbia.edu/data/video_and_anns.tar.gz"
OOPS_MAPPING_FILE = "data_files/oops_video_mapping.csv"
EXPECTED_OOPS_COUNT = 818
EXPECTED_SYN_COUNT = 12000

# 16 activity classes shared across all components
ACTIVITY_LABELS = [
    "walk",         # 0
    "fall",         # 1
    "fallen",       # 2
    "sit_down",     # 3
    "sitting",      # 4
    "lie_down",     # 5
    "lying",        # 6
    "stand_up",     # 7
    "standing",     # 8
    "other",        # 9
    "kneel_down",   # 10
    "kneeling",     # 11
    "squat_down",   # 12
    "squatting",    # 13
    "crawl",        # 14
    "jump",         # 15
]

# --- Config categorization for video loading ---

# Configs that have OF-Syn videos (same-domain syn configs)
SYN_VIDEO_CONFIGS = {
    "of-syn",
    "of-syn-cross-age",
    "of-syn-cross-ethnicity",
    "of-syn-cross-bmi",
}

# Configs where OOPS videos are the only video source
OOPS_VIDEO_CONFIGS = {
    "of-itw",
}

# "to-all" configs: test from ALL datasets (staged + itw + syn).
# Syn train/val, test from all (need syn for train+test, OOPS for test)
TO_ALL_SYN_TRAIN_CONFIGS = {
    "of-syn-to-all-cs",
    "of-syn-to-all-cv",
}

# Staged train/val (no downloadable video for train/val),
# test from all (need OOPS + syn for test)
TO_ALL_STAGED_TRAIN_CONFIGS = {
    "of-sta-to-all-cs",
    "of-sta-to-all-cv",
    "caucafall-to-all-cs", "caucafall-to-all-cv",
    "cmdfall-to-all-cs", "cmdfall-to-all-cv",
    "edf-to-all-cs", "edf-to-all-cv",
    "gmdcsa24-to-all-cs", "gmdcsa24-to-all-cv",
    "le2i-to-all-cs", "le2i-to-all-cv",
    "mcfd-to-all-cs", "mcfd-to-all-cv",
    "occu-to-all-cs", "occu-to-all-cv",
    "up_fall-to-all-cs", "up_fall-to-all-cv",
}

# Staged+syn train/val, test from all (need syn for train+test, OOPS for test)
TO_ALL_STAGED_SYN_TRAIN_CONFIGS = {
    "of-sta-syn-to-all-cs",
    "of-sta-syn-to-all-cv",
}

# Staged-only configs (no downloadable video, but usable with OMNIFALL_ROOT)
STAGED_ONLY_CONFIGS = {
    "of-sta-cs", "of-sta-cv",
    "cs", "cv",
    "caucafall-cs", "cmdfall-cs", "edf-cs", "gmdcsa24-cs",
    "le2i-cs", "mcfd-cs", "occu-cs", "up_fall-cs",
    "caucafall-cv", "cmdfall-cv", "edf-cv", "gmdcsa24-cv",
    "le2i-cv", "mcfd-cv", "occu-cv", "up_fall-cv",
}

# Configs that support video loading when downloading (no OMNIFALL_ROOT)
DOWNLOADABLE_VIDEO_CONFIGS = (
    SYN_VIDEO_CONFIGS | OOPS_VIDEO_CONFIGS
    | TO_ALL_SYN_TRAIN_CONFIGS | TO_ALL_STAGED_TRAIN_CONFIGS
    | TO_ALL_STAGED_SYN_TRAIN_CONFIGS
)

# All configs that support video loading with OMNIFALL_ROOT
ALL_VIDEO_CONFIGS = DOWNLOADABLE_VIDEO_CONFIGS | STAGED_ONLY_CONFIGS

# Configs with no video source under any circumstances (metadata only)
NO_VIDEO_CONFIGS = {
    "labels", "labels-syn", "metadata-syn", "framewise-syn",
}

OOPS_LICENSE_TEXT = """\
==========================================================================
OOPS Dataset License Notice
==========================================================================

The OF-ItW component of OmniFall uses videos from the OOPS dataset.
The following notice is from the OOPS dataset website
(https://oops.cs.columbia.edu/data/):

  "By pressing any of the links above, you acknowledge that we do not
   own the copyright to these videos and that they are solely provided
   for non-commercial research and/or educational purposes. This dataset
   is licensed under a Creative Commons Attribution-NonCommercial-
   ShareAlike 4.0 International License."

If you use OF-ItW in your research, please also cite the OOPS paper:

  @inproceedings{epstein2020oops,
    title={Oops! predicting unintentional action in video},
    author={Epstein, Dave and Chen, Boyuan and Vondrick, Carl},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition},
    pages={919--929},
    year={2020}
  }

The download will stream ~45GB from the OOPS website and extract
%d videos (~2.6GB disk space).
=========================================================================="""
