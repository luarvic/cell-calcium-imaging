# Cell Calcium Imaging

This project uses https://github.com/flatironinstitute/CaImAn

To run the scripts you need first to activate `conda`:

```zsh
conda activate caiman
```

Next, to run Jupyter notebook:

```zsh
jupyter notebook
```

To run the analysis in `Jupyter` execute these:

```
# Generate maps for all tiffs and then postprocess (scan all results under --outroot)
!python caiman_pipeline.py --inputs ./tiffs --fr 4 --outroot results
!python postprocess.py --outroot results --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf

# Generate maps for well1-000.tif and then postprocess (single results folder)
!python caiman_pipeline.py --inputs ./tiffs/well1-000.tif --fr 4 --outroot results
!python postprocess.py --results results/well1-000 --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf

# Optional knobs:
# --plots_dir: subfolder name inside each results/<tiff>/
# --thr_frac: ROI threshold used for the labeled ROI map (higher = smaller labels/cores)
```
