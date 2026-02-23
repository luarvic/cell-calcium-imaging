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

Place `.tif` files under 'tiffs' directory in the project root directory.

To run the analysis in Jupyter, execute:

```
# Generate maps for all TIFFs, then postprocess (scan all results under --outroot)
!python caiman_pipeline.py --inputs ./tiffs --fr 4 --outroot results
!python postprocess.py --outroot results --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf --dedup --dedup_temporal_thr 0.8 --dedup_ignore_dist

# Generate maps for well1-000.tif, then postprocess (single results folder)
!python caiman_pipeline.py --inputs ./tiffs/well1-000.tif --fr 4 --outroot results
!python postprocess.py --results results/well1-000 --fr 4 --plots_dir plots --thr_frac 0.2 --zscore --heatmap --pdf --dedup --dedup_temporal_thr 0.8 --dedup_ignore_dist

# New cell-level postprocess (postprocess_cells.py)
# - Runs deduplicate_rois_cellify.py (dedup) and can optionally collapse soma+process ROIs into one cell-level ROI per cell.
# - Writes *_cells outputs and cell_map.json when --cellify is used.
# - Uses map_neurons_cells.py for ROI maps (cell-level when --cellify, dedup-level otherwise).

# Run for all TIFFs (scan all results under --outroot)
!python postprocess_cells.py --outroot results --fr 4 --plots_dir plots --thr_frac 0.2 --dedup --dedup_temporal_thr 0.8 --dedup_ignore_dist --cellify --soma_nms_dist 25

# Run for a single TIFF (single results folder)
!python postprocess_cells.py --results results/well1-000 --fr 4 --plots_dir plots --thr_frac 0.2 --dedup --dedup_temporal_thr 0.8 --dedup_ignore_dist --cellify --soma_nms_dist 25

# Optional knobs:
# --plots_dir: subfolder name inside each results/<tiff>/
# --thr_frac: ROI threshold used for the labeled ROI map (higher = smaller labels/cores)
```
