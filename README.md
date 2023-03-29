# These modules are highly customized for my own research project
`load.py` contains functions that load my datasets. unlikely to be useful for other projects

`compute.py` contains a few useful functions that might be adapted for other projects, e.g.,
- **multitaper_spectrogram**: using dpss. the average spectrogram is weighted with eigen values
- **pev**: percentage of explained variance. measured in $\omega^2$
- **f1_score**: using svm as decoding algorithm
