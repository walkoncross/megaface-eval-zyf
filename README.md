# MegaFace Evaluation and Plot

Please refer to [MegaFace's Official Page](http://megaface.cs.washington.edu/participate/challenge.html) if you have any question about the evaluation/challenge.

## Evaluation

1. Download Megaface [Linux Development Kit](http://megaface.cs.washington.edu/participate/challenge.html) ([.zip](http://megaface.cs.washington.edu/dataset/download/content/devkit.zip) or [.tar.gz](http://megaface.cs.washington.edu/dataset/download/content/devkit.tar.gz)) and unzip/untar it to ./devkit.

1. Run the evaluation according to the [devkit/readme.txt](http://megaface.cs.washington.edu/participate/content/readme.txt).

## Plot results

1. Download the some of the [Challenge 1 json results](http://megaface.cs.washington.edu/Challenge1JSON.zip), and unzip it into devkit/Challenge1External;

1. For plot test, run:
```
cd plot_results
python plot_megaface_results.py
```

1. To plot your own evaluation results, please change "method_dirs" and "method_labels" to your results's paths in [plot_results/plot_megaface_results.py](plot_results/plot_megaface_results.py), and run the script.

