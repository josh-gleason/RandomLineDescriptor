#!/bin/bash

bases=("graf" "bikes" "ubc" "leuven" "trees" "wall" "boat" "bark")
exts=("ppm" "ppm" "ppm" "ppm" "ppm" "ppm" "pgm" "ppm")
#bases=("graf")
#exts=("ppm")
for i in ${!bases[*]}
do
   for n in 2 3 4 5 6
   do
      REFIMG="../TestImages/${bases[i]}/img1.${exts[i]}"
      MATIMG="../TestImages/${bases[i]}/img${n}.${exts[i]}"
      HOMOGRAPHY="../TestImages/${bases[i]}/H1to${n}p"

      TITLE="Precision-Recall ${bases[i]}${n}"
      OUTPUT_PDF="${bases[i]}${n}_PR.png"

      ROCTIT="ROC Curve ${bases[i]}${n}"
      ROC_PDF="${bases[i]}${n}_ROC.png"

      ./bin/tester -c ./bin/config.cfg --SHOW_IMG=0 --SAVE_IMG=0 --SAVE_CONFIG=0 --REF_IMG=${REFIMG} --MAT_IMG=${MATIMG} --HOMOG=${HOMOGRAPHY} && octave --silent --eval "plotter(\"PR.csv\",\"${TITLE}\",\"${OUTPUT_PDF}\",\"${ROCTIT}\",\"${ROC_PDF}\")"
   done
done


