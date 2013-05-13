#ifndef RESULTS_H
#define RESULTS_H

struct Results {
   double accuracy;
   int correctMatches;
   int incorrectMatches;
   int correspondences;
   clock_t trainingTime;
   clock_t refDetectTime;
   clock_t matchDetectTime;
   clock_t matchingTime;
};

#endif

