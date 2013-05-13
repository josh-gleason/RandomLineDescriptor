#ifndef RESULTS_H
#define RESULTS_H

struct Results {
   double accuracy;
   double tpr; // true positive rate (recall)
   double fpr; // false positive rate (inverse recall)
   int correctMatches;  // true positive
   int incorrectMatches;   // false positive
   int trueNeg;
   int falseNeg;
   clock_t trainingTime;
   clock_t refDetectTime;
   clock_t matchDetectTime;
   clock_t matchingTime;
};

#endif

