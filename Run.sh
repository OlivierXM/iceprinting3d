#!/bin/bash
outputDir=./Outputs$2
utilityDir=./
rm -rf $outputDir
fileName=Wall_2D_additive.py
# mpiexec -n $1 python3 BlockFreeze.py $dfxtools $utilityDir "Outputs"$2"/"
mpiexec --n $1 -cpu-set $3 --bind-to core --report-bindings python3 $fileName $dfxtools $utilityDir "Outputs"$2"/"
# python3 `script` `path/to/dolfinxtools/Tools` `path/to/iceprinting3d/Tools` `Output/`