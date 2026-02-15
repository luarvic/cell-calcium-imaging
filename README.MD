# Cell Calcium Imaging

This project uses https://github.com/flatironinstitute/CaImAn.

To run the scripts, first activate `conda`:

```zsh
conda activate caiman
```

Next, start Jupyter Notebook:

```zsh
jupyter notebook
```

To run the analysis in Jupyter, execute:

```
# Generate maps for all TIFFs, then postprocess (scan all results under --outroot)
!python caiman_pipeline.py --inputs ./tiffs --fr 4 --outroot results
!python postprocess.py --outroot results --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf

# Generate maps for well1-000.tif, then postprocess (single results folder)
!python caiman_pipeline.py --inputs ./tiffs/well1-000.tif --fr 4 --outroot results
!python postprocess.py --results results/well1-000 --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf

# Optional knobs:
# --plots_dir: subfolder name inside each results/<tiff>/
# --thr_frac: ROI threshold used for the labeled ROI map (higher = smaller labels/cores)
```
