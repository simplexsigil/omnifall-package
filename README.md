# omnifall

Companion package for the [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall)
fall-detection dataset.

OmniFall puts ten video datasets into one taxonomy of 16 classes. The
annotations are on the HuggingFace Hub. The videos of eight components belong to
other authors, and those authors do not permit redistribution. This package
connects the two halves: it reads the annotations from the Hub, it obtains the
videos from their original sources, and it gives you PyTorch datasets that
decode them.

> This document uses Simplified Technical English (ASD-STE100). Sentences are
> short. Each instruction is one step. One word has one meaning.

---

## 1. Installation

Install the package with `pip`:

```bash
pip install omnifall                  # annotations only
pip install 'omnifall[video]'         # add video decoding (PyTorch, PyAV)
pip install 'omnifall[transformers]'  # add HuggingFace model support
```

The basic installation needs only `datasets` and `huggingface-hub`. It does not
import PyTorch. Use it when you want the annotations and not the videos.

To convert videos from their original releases, you must also install `ffmpeg`.
The package finds `ffmpeg` on the `PATH`. If it is not on the `PATH`, set the
`OMNIFALL_FFMPEG` variable to the program or to its directory.

---

## 2. Concepts

Read this section first. It explains four terms that this document uses many
times.

**Component.** One of the ten source datasets. The ten components are
`caucafall`, `cmdfall`, `edf`, `GMDCSA24`, `le2i`, `mcfd`, `occu`, `up_fall`,
`OOPS` and `of-syn`.

**Config.** One way to divide the data. A config selects components and applies
a split. `le2i-cs` is one component with a cross-subject split. `cs` is all
staged components with a cross-subject split. The Hub serves 72 configs.

**Segment.** One row of a config. A segment is a part of a video with one label,
a start time and an end time. One video usually contains many segments.

**Path.** The `path` column of a segment. The value is relative to its own
component, and it has no file extension. To find the file, join the component
and the path:

```
{root}/{dataset}/video/{path}.mp4
```

Every config has a `dataset` column, so this rule applies to all of them. Do not
use the config name to find a file.

---

## 3. Read the annotations

The annotations are small. This step needs no videos.

```python
import omnifall

parts = omnifall.load("le2i-cs")            # all splits
train = omnifall.load("le2i-cs", split="train")
names = omnifall.list_configs()             # the 72 configs on the Hub
```

`load()` gives you a `datasets.Dataset` or a `datasets.DatasetDict`. All
operations of the `datasets` library apply to it.

Each row has these columns:

| Column | Type | Meaning |
|---|---|---|
| `path` | string | video path, relative to its component, without extension |
| `dataset` | string | the component the video belongs to |
| `label` | int | class id from 0 to 15, as a `ClassLabel` |
| `start` | float | start of the segment, in seconds |
| `end` | float | end of the segment, in seconds |
| `subject` | int | subject id, or `-1` if the component has none |
| `cam` | int | camera id, or `-1` if the component has none |

Configs of OF-Syn add 12 more columns with demographic and scene metadata.

Two configs are different. `metadata-syn` describes whole videos.
`framewise-syn` holds one label for each frame. Neither has segment times, so
neither can have a video clip.

---

## 4. Get the videos

### 4.1 If you already have the videos

Set `OMNIFALL_ROOT` to the directory that holds them:

```bash
export OMNIFALL_ROOT=/path/to/data     # {root}/{dataset}/video/{path}.mp4
```

To use a different directory for one component, set its own variable. This
variable has priority over `OMNIFALL_ROOT`:

```bash
export OMNIFALL_VIDEO_ROOT__le2i=/somewhere/else/le2i/video
```

### 4.2 If you do not have the videos

Nine components download without an account, an API key or a browser:

```bash
omnifall status              # show what each component needs
omnifall prepare of-syn      # download one component
omnifall prepare --all       # download all that can be downloaded
omnifall verify le2i         # compare a directory with the annotations
```

`prepare` does three things. It downloads the original release. It converts the
videos to the OmniFall layout. It then verifies the result against the
annotations on the Hub, and it stops with an error if a file is missing.

CMDFall is the tenth component. You cannot download it. Its authors give access
after an e-mail request. Read section 4.4.

### 4.3 Sizes and times

| Component | Download | Note |
|---|---|---|
| `of-syn` | 9.0 GiB | one archive from the Hub |
| `OOPS` | 44.6 GiB | streamed; only 2.6 GiB is written to disk |
| `edf` | 14.9 GiB | Zenodo |
| `occu` | 10.1 GiB | Zenodo |
| `le2i` | 8.9 GiB | one archive that holds six more |
| `caucafall` | 7.8 GiB | Mendeley |
| `mcfd` | 3.5 GiB | 24 archives, one for each scenario |
| `GMDCSA24` | 1.0 GiB | GitHub |
| `up_fall` | large | 1,118 archives, one for each trial and camera |
| `cmdfall` | — | by e-mail request only |

The download of OOPS takes about 30 minutes. The conversion of `up_fall` takes
much longer, because the package builds each video from its frames.

### 4.4 To download a component yourself

You do not have to let the package reach a source. Use this method when the
machine has no network, when a site needs a browser, or when you already have
the archive.

Step 1. Read what to download and where to put it:

```bash
omnifall sources mcfd
```

The command prints the file names, the destination directory, and the terms of
the authors.

Step 2. Put the archives in the download directory. Do not rename them:

```bash
export OMNIFALL_DOWNLOAD_DIR=/data/omnifall-downloads   # default: {cache}/downloads
```

Step 3. Run `prepare`. It finds the archives and does not download them:

```bash
omnifall prepare mcfd
```

You can also give a location directly. The argument accepts one archive, a
directory of archives, or a release that you unpacked:

```bash
omnifall prepare caucafall --archive ~/Downloads/'Dataset CAUCAFall.zip'
omnifall prepare mcfd --archive /data/mcfd-zips/ --workers 8
omnifall convert le2i /data/Le2i-unpacked/
```

An incomplete set is permitted. The package uses the archives that are present,
and it downloads only the others.

---

## 5. Load videos

Add a `video` column that holds absolute file paths:

```python
ds = omnifall.load("le2i-cs", split="train", video=True)
ds[0]["video"]      # '/path/to/le2i/video/Coffee_room_01/video_1.mp4'
```

If a file is absent, its `video` value is `None`, and you get one warning for
the whole dataset. In a training script you usually want an immediate failure
instead:

```python
ds = omnifall.load("cs", video=True, strict=True)   # raises MissingVideosError
```

To examine a dataset without a change to it, ask for a report:

```python
print(omnifall.resolution_report(omnifall.load("cs")).summary())
```

---

## 6. Train a model

### 6.1 PyTorch

```python
import omnifall
from torch.utils.data import DataLoader

parts = omnifall.load_video_dataset("le2i-cs", num_frames=16, target_fps=15)
loader = DataLoader(parts["train"], batch_size=8, collate_fn=omnifall.collate_fn)

batch = next(iter(loader))
batch["pixel_values"].shape    # (8, 16, 3, 224, 224)
batch["labels"].shape          # (8,)
```

The dataset decodes only the segment of each row. It does not decode the whole
file.

### 6.2 Temporal sampling

The `sampling` argument selects the frames inside a segment:

| Value | Behaviour | Use it for |
|---|---|---|
| `"random"` | a window at a random position | training |
| `"uniform"` | frames spread over the whole segment | evaluation |
| `"center"` | a window in the middle of the segment | evaluation |
| `"auto"` | `"random"` for the train split, `"uniform"` for the others | the default |

`"uniform"` and `"center"` give the same frames at each call. `"random"` gives
the same frames if you set `seed`. Evaluation results are therefore repeatable.

### 6.3 Tensor layout

| Value of `output_format` | Shape of one sample | Shape of one batch |
|---|---|---|
| `"TCHW"` (default) | `(T, C, H, W)` | `(B, T, C, H, W)` |
| `"CTHW"` | `(C, T, H, W)` | `(B, C, T, H, W)` |
| `"THWC"` | `(T, H, W, C)`, uint8 | raw frames; no transform permitted |

`"TCHW"` is the layout that HuggingFace video models accept. Do not permute it.

### 6.4 HuggingFace transformers

```python
from transformers import Trainer, TrainingArguments

parts = omnifall.trainer_dataset("le2i-cs", model_name="MCG-NJU/videomae-base")
trainer = Trainer(
    model=omnifall.load_model("MCG-NJU/videomae-base"),
    train_dataset=parts["train"],
    eval_dataset=parts["validation"],
    data_collator=omnifall.collate_fn,
    compute_metrics=omnifall.compute_metrics,
    args=TrainingArguments(output_dir="out", remove_unused_columns=False),
)
trainer.train()
```

Set `remove_unused_columns=False`. If you do not, the `Trainer` removes the
columns that the dataset needs.

`load_model()` changes the classifier to 16 outputs. The new classifier has
random weights. This is correct, and you must train it.

`compute_metrics()` reports accuracy and balanced accuracy. OmniFall has many
more `other` segments than `fall` segments. A model that never predicts `fall`
can still get a high accuracy, so use the balanced accuracy.

---

## 7. Errors

The package raises three different errors for three different problems. Do not
confuse them.

| Error | Cause | What to do |
|---|---|---|
| `MissingVideosError` | files are absent | run `omnifall prepare`, or set `OMNIFALL_ROOT` |
| `VideoUnavailableError` | one row has no file | the same |
| `VideoDecodeError` | the file is present but damaged | download the component again |

`MissingVideosError` and `VideoUnavailableError` are subclasses of
`FileNotFoundError`. `VideoDecodeError` is not, because the file exists.

---

## 8. Environment variables

| Variable | Effect |
|---|---|
| `OMNIFALL_ROOT` | the video root, `{root}/{dataset}/video/{path}.mp4` |
| `OMNIFALL_VIDEO_ROOT__<dataset>` | the video directory of one component |
| `OMNIFALL_CACHE_DIR` | where prepared videos go (default `~/.cache/omnifall`) |
| `OMNIFALL_DOWNLOAD_DIR` | where original releases go (default `{cache}/downloads`) |
| `OMNIFALL_FFMPEG` | the `ffmpeg` program, or its directory |

An explicit argument has priority over all variables. A per-component variable
has priority over `OMNIFALL_ROOT`. `OMNIFALL_ROOT` has priority over the cache.

---

## 9. Command-line interface

| Command | Function |
|---|---|
| `omnifall info` | show the cache directories and the components on disk |
| `omnifall status` | show what each component needs |
| `omnifall configs` | list the configs on the Hub |
| `omnifall sources <ds>` | show the origin, the terms and the citation |
| `omnifall prepare <ds>` | download and convert a component |
| `omnifall convert <ds> <dir>` | convert a release that you unpacked |
| `omnifall verify <ds>` | compare a directory with the annotations |
| `omnifall cite <ds>` | print the BibTeX that you must cite |

Add `--all` to apply a command to all components. Add `--workers N` to set the
number of parallel jobs.

---

## 10. Examples

The `examples/` directory holds three notebooks. Each one runs from start to
end.

| Notebook | Content |
|---|---|
| [01_quickstart.ipynb](examples/01_quickstart.ipynb) | configs, labels, annotations; no download |
| [02_videos.ipynb](examples/02_videos.ipynb) | how to get videos and decode a segment |
| [03_training.ipynb](examples/03_training.ipynb) | dataloaders, transformers, cross-domain tests |

---

## 11. Known behaviour

**Some segments end after their video ends.** In `edf`, `cmdfall` and `mcfd`,
209 of 52,618 segments continue past the last frame. The cause is in the
annotations, not in the video files: these components annotate one camera view
and move the result to the other views by a time offset, and the offset is not
limited to the length of the target video. The package limits each segment to
the available frames and repeats the last frame. It does not raise an error,
because that would reject correct data.

**Some component names are aliases.** Older config names still work. Use
`omnifall.DEPRECATED_CONFIGS` to see the current name of each one. A bare
component name such as `le2i` is an alias for `le2i-cs`, so it gives you the
cross-subject split. Ask for `le2i-cv` if you want the cross-view split.

**CMDFall gives more videos than OmniFall uses.** The release holds 1,436
videos. OmniFall annotates the 384 continuous recordings and not the 1,052
short clips. Files that the annotations do not name are correct and expected.

For more detail, read
[KNOWN_PITFALLS.md](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/KNOWN_PITFALLS.md)
on the dataset page.

---

## 12. Citation

OmniFall contains the work of other research groups. Cite the OmniFall paper and
also the paper of each component that you use. This command prints the correct
BibTeX for a config:

```bash
omnifall cite le2i-cs
```

Some components give no license. They ask only for a citation. Run
`omnifall sources <ds>` to read the words of their authors.

```bibtex
@misc{omnifall,
      title={OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection},
      author={David Schneider and Zdravko Marinov and Rafael Baur and Zeyun Zhong and Rodi Düger and Rainer Stiefelhagen},
      year={2025},
      eprint={2505.19889},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.19889},
}
```

---

## 13. License

The code of this package is under the Apache License 2.0. See [LICENSE](LICENSE).

This license does not apply to the dataset. The annotations, and the videos of
each component, keep the terms of their own authors.

---

## 14. Links

- [HuggingFace dataset](https://huggingface.co/datasets/simplexsigil2/omnifall)
- [Paper](https://arxiv.org/abs/2505.19889)
- [Project page](https://simplexsigil.github.io/omnifall/)
