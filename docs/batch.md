# Batch runs

`scripts/run_batch.py` fits every cluster in a member table:

```bash
python scripts/run_batch.py members.parquet --grid isochrones/ \
    --priors priors.ecsv --outdir run/ --chunk 0 --total-chunks 64 --workers 56
```

* **Member table:** one row per star, with a cluster-name column
  (`--name-col`, default `name`).
* **Chunks:** clusters are sorted by name and split into `--total-chunks`
  equal chunks.
* **Restartable:** each cluster's chain goes to `run/chains/<name>_ns_ucc.h5`.
  Existing chains are skipped, so an interrupted job can simply be resubmitted.
* **Per-chunk summary:** `run/results_chunkNNNN.ecsv` (astropy ECSV, with units).
  Merge the chunks with `astropy.table.vstack`.

On a cluster, submit one job per chunk (e.g. a SLURM array with
`--chunk $SLURM_ARRAY_TASK_ID`). The published catalogue (5056 clusters) was run as 64 chunks of 79 clusters on the TRUBA
cluster (TÜBİTAK ULAKBIM).
